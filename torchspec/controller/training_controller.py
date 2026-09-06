# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Async Training Controller for decoupled inference and training.

Data flow:
  load_dataset(args) → Stored Dataset → Prompt Buffer → Inference Manager
    → Sample Pool → Train Queues → Training Workers
  (Stored dataset is retained for epoch reloads and vocab mapping computation.)

Controller manages the tokenized dataset (for epoch reloads and vocab mapping),
prompt metadata, and mooncake keys. Actual inference tensor data is stored in
mooncake; the controller only tracks keys and byte sizes for backpressure.

Batch Size Design:
  redis.dispatch_batch_size          # Samples published per XADD pipeline round trip

  This is now purely publish granularity. The DP fan-out that used to derive it
  (micro_batch_size -> per_dp_rank_batch_size -> dispatch_batch_size) lives on
  the consumer side: the controller publishes one flat stream and knows nothing
  about how many trainers read it or at what data-parallel degree.
"""

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field
import torch
import json
from typing import Any, Dict, Optional, Tuple
import redis
import os

from torchspec.data.utils import length_grouped_order
from torchspec.utils.logging import logger
from torchspec.utils.memory import estimate_tensor_bytes
from torchspec.utils.types import InferenceInput, InferenceOutput
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

_estimate_bytes = estimate_tensor_bytes


@dataclass
class SpeedMonitor:
    """Tracks throughput over a sliding time window."""

    window_seconds: float = 10.0
    _events: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_count: int = 0

    def record(self, count: int = 1) -> None:
        """Record count entries at current time."""
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._total_count += count
            self._prune_old_events(now)

    def _prune_old_events(self, now: float) -> None:
        """Remove events outside the window."""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def get_speed(self) -> float:
        """Get current speed in entries/sec over the window."""
        now = time.time()
        with self._lock:
            self._prune_old_events(now)
            if not self._events:
                return 0.0

            window_count = sum(count for _, count in self._events)
            oldest_time = self._events[0][0]
            elapsed = now - oldest_time
            if elapsed < 0.001:
                return 0.0
            return window_count / elapsed

    def get_total_count(self) -> int:
        """Get total count since start."""
        return self._total_count


@dataclass
class MooncakeEntry:
    inserted_at: float
    dispatched_at: float | None
    num_bytes: int
    has_last_hidden_states: bool
    has_target: bool

# Wire-format version for the Redis stream. Bump on any breaking change to the
# field set; consumers must check it before parsing. Every entry carries it.
STREAM_SCHEMA_VERSION = "1"
STREAM_TYPE_SAMPLE = "sample"
STREAM_TYPE_EOS = "eos"
STREAM_TYPE_HEARTBEAT = "heartbeat"


def eos_fields() -> dict[str, str]:
    """Terminal entry. Consumers stop when they read this."""
    return {"v": STREAM_SCHEMA_VERSION, "type": STREAM_TYPE_EOS}


# Run-metadata hash field values.
RUN_STATUS_RUNNING = "running"
RUN_STATUS_FINISHED = "finished"


def heartbeat_fields(producer_idx: int, epoch: int) -> dict[str, str]:
    """Liveness ping so a consumer can tell "producer slow" from "producer gone".

    ``epoch`` is the producer's *current* pass over the dataset, not a label for
    any particular sample -- a sample published now may have been queued during
    the previous pass. Treat it as a coarse progress signal.
    """
    return {
        "v": STREAM_SCHEMA_VERSION,
        "type": STREAM_TYPE_HEARTBEAT,
        "sent_at": str(time.time()),
        "producer_idx": str(producer_idx),
        "epoch": str(epoch),
    }


@dataclass
class TrainSampleRedis:
    """One training sample's metadata, as it crosses the Redis stream.

    Every entry carries ``v`` and ``type``; consumers must branch on ``type``
    before parsing, since control entries (eos, heartbeat) share the stream and
    carry none of the sample fields.
    """
    mooncake_key: str
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Optional[Dict[str, torch.dtype]] = None
    packed_loss_mask: Optional[str] = None
    expires_at: Optional[float] = None
    producer_idx: Optional[int] = None
    last_turn_loss_only: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    data_id: Optional[str] = None

    def to_redis_fields(self) -> dict[str, str]:
        fields = {
            "v": STREAM_SCHEMA_VERSION,
            "type": STREAM_TYPE_SAMPLE,
            "mooncake_key": self.mooncake_key,
            "tensor_shapes": json.dumps(self.tensor_shapes),
        }

        if self.tensor_dtypes is not None:
            fields["tensor_dtypes"] = json.dumps({
                name: str(dtype).removeprefix("torch.")
                for name, dtype in self.tensor_dtypes.items()
            })

        if self.packed_loss_mask is not None:
            fields["packed_loss_mask"] = self.packed_loss_mask

        if self.expires_at is not None:
            fields["expires_at"] = str(self.expires_at)
        else:
            raise ValueError("expires_at is required but not set for mooncake_key: " + self.mooncake_key)

        if self.producer_idx is not None:
            fields["producer_idx"] = str(self.producer_idx)

        if self.last_turn_loss_only is not None:
            fields["last_turn_loss_only"] = (
                "1" if self.last_turn_loss_only else "0"
            )

        # Consumers should not rely on this field
        # It's an artifact of the previous implementation
        if self.metadata is not None:
            fields["metadata"] = json.dumps(self.metadata)

        if self.data_id is not None:
            fields["data_id"] = self.data_id

        return fields

    @classmethod
    def from_redis_fields(cls, fields: dict) -> "TrainSampleRedis":
        """Parse a stream entry. Consumer-side only.

        Raises ValueError on an unknown schema version or a non-sample entry,
        so a consumer that forgets to branch on ``type`` fails loudly rather
        than KeyError-ing on a missing ``mooncake_key``.
        """
        version = fields.get("v")
        if version != STREAM_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported stream schema version {version!r} "
                f"(this consumer understands {STREAM_SCHEMA_VERSION!r})"
            )
        entry_type = fields.get("type")
        if entry_type != STREAM_TYPE_SAMPLE:
            raise ValueError(
                f"Not a sample entry (type={entry_type!r}); branch on 'type' before parsing"
            )
        tensor_shapes = {
            name: tuple(shape)
            for name, shape in json.loads(
                fields["tensor_shapes"]
            ).items()
        }

        tensor_dtypes = None
        if "tensor_dtypes" in fields:
            tensor_dtypes = {
                name: getattr(torch, dtype_name)
                for name, dtype_name in json.loads(
                    fields["tensor_dtypes"]
                ).items()
            }

        metadata = None
        if "metadata" in fields:
            metadata = json.loads(fields["metadata"])

        last_turn_loss_only = None
        if "last_turn_loss_only" in fields:
            last_turn_loss_only = fields["last_turn_loss_only"] == "1"

        return cls(
            mooncake_key=fields["mooncake_key"],
            tensor_shapes=tensor_shapes,
            tensor_dtypes=tensor_dtypes,
            packed_loss_mask=fields.get("packed_loss_mask"),
            expires_at=float(fields.get("expires_at", 0.0)),
            producer_idx=int(fields.get("producer_idx", 0)),
            last_turn_loss_only=last_turn_loss_only,
            metadata=metadata,
            data_id=fields.get("data_id"),
        )


class AsyncTrainingController:
    """Central controller for async training pipeline.

    Responsibilities:
      - Loads and stores tokenized datasets for training and eval
      - Computes vocab mappings for draft model pruning
      - Manages prompt buffer (samples waiting for inference)
      - Manages sample pool (completed inferences waiting for training)
      - Dispatches batches to per-DP training queues when pool is full
      - Monitors inference and training throughput
    """

    def __init__(self, args, mooncake_store: EagleMooncakeStore = None):
        self.args = args

        self.prompt_buffer: deque[InferenceInput] = deque()
        self._prompt_lock = threading.Lock()

        self.sample_pool: deque[InferenceOutput] = deque()
        self._pool_lock = threading.Lock()
        self._pool_bytes = 0
        self._sample_bytes: dict[str, int] = {}
        self._mooncake_store = mooncake_store
        self._mooncake_entries: dict[str, MooncakeEntry] = {}
        self._mooncake_bytes = 0
        redis_host = os.environ["REDIS_HOST"]
        redis_port = os.environ["REDIS_PORT"]
        self.redis_train_stream = getattr(args, "redis_train_stream", "train_samples")
        self.redis_eval_stream = getattr(args, "redis_eval_stream", "eval_samples")
        self._redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0)
        self._publish_max_attempts = getattr(args, "redis_publish_max_attempts", 2)
        self._publish_retry_seconds = getattr(args, "redis_publish_retry_seconds", 2)
        self._stream_maxlen = getattr(args, "redis_stream_maxlen", 16384)
        self._meta_key = (
            f"{self.redis_train_stream}:"
            f"{getattr(args, 'redis_meta_key_suffix', 'meta')}"
        )
        self._min_expected_seq_len = getattr(args, "redis_min_expected_seq_len", 128)
        self._heartbeat_seconds = getattr(args, "redis_heartbeat_seconds", 10.0)
        self._last_heartbeat = 0.0
        self._producer_idx = 0
        self._publish_dropped = 0
        self._publish_failures = 0
        self._check_stream_maxlen()

        # Eval: separate pool and queues so eval data never mixes with training
        # Eval is currently unused and requires additional code modifications to work correctly.
        self.eval_pool: deque[InferenceOutput] = deque()
        self._eval_pool_lock = threading.Lock()
        self._eval_data_ids: set[str] = set()
        self._eval_expected_count: int = 0
        self._eval_dispatched_samples: int = 0

        self.batch_id = 0
        self.dispatch_batch_size = getattr(args, "redis_dispatch_batch_size", 1)
        self.eval_dispatch_batch_size = None
        self._data_id_counter = 0

        self._stored_dataset: list | None = None
        self._stored_eval_dataset: list | None = None
        self._dataset_epoch: int = 0
        self._dataset_seed: int = getattr(args, "seed", 42)
        self._shuffle_dataset: bool = getattr(args, "shuffle_dataset", True)
        self._length_group_size: int = getattr(args, "length_group_size", 32)

        self._start_time = time.time()
        self._inference_monitor = SpeedMonitor(window_seconds=10.0)
        self._training_monitor = SpeedMonitor(window_seconds=10.0)
        self._last_dispatch_log_time = 0.0
        self._inference_error: str | None = None
        self._consecutive_errors: int = 0
        self._last_error_log_time = 0.0
        self._error_lock = threading.Lock()
        self._eviction_ttl = getattr(args, "mooncake_sample_eviction_ttl_seconds", 60.0)
        self._eviction_interval = getattr(args, "mooncake_sample_eviction_check_interval", 1.0)
        self._eviction_stop = threading.Event()
        self._eviction_thread: threading.Thread | None = None
        self.verified_tensor_shapes = False


    def _check_stream_maxlen(self) -> None:
        """Fail if the stream cannot hold a reference to every resident sample.

        The stream must retain an entry at least as long as Mooncake retains
        the tensors it points at, or consumers never learn about samples that
        are still fetchable. Working that through, the TTL cancels out:

            N >= rate * ttl,  rate <= watermark * segment / (ttl * bytes)
            =>  N >= watermark * segment / bytes_per_sample

        i.e. the stream must be at least as long as the maximum number of
        samples that can be resident at once. Short samples are cheap, so more
        of them fit -- which is why this is sized against the *shortest*
        expected sequence, not the average.
        """
        from torchspec.config.mooncake_config import MooncakeConfig

        segment = getattr(self.args, "mooncake_global_segment_size", None)
        hidden_dim = getattr(self.args, "mooncake_hidden_dim", None)
        num_aux = getattr(self.args, "mooncake_num_aux_layers", None)
        watermark = getattr(self.args, "mooncake_watermark_fraction", 0.75)
        if segment is None or hidden_dim is None or num_aux is None:
            logger.warning("Cannot check redis.stream_maxlen: mooncake sizing config missing")
            return

        if isinstance(segment, str):
            segment = MooncakeConfig.parse_size(segment)
        # hidden_states + last_hidden_states (bf16) + input_ids (int64), per token
        bytes_per_token = (num_aux * hidden_dim + hidden_dim) * 2 + 8
        min_sample_bytes = self._min_expected_seq_len * bytes_per_token
        required = int(watermark * segment / min_sample_bytes) + 1

        logger.info(
            "redis.stream_maxlen=%d (>= %d required: watermark=%.2f x segment=%.1fGiB / "
            "%.1fMiB per %d-token sample)",
            self._stream_maxlen, required, watermark, segment / 1024**3,
            min_sample_bytes / 1024**2, self._min_expected_seq_len,
        )
        if self._stream_maxlen < required:
            raise ValueError(
                f"redis.stream_maxlen={self._stream_maxlen} is below the {required} entries "
                f"needed to reference every sample Mooncake can hold at once. Raise it, or "
                f"raise redis.min_expected_seq_len if {self._min_expected_seq_len} tokens is "
                f"shorter than anything your corpus actually produces."
            )

    def publish_run_meta(self, dataset_size: int) -> dict[str, str]:
        """Publish the run plan to ``{stream}:meta`` for consumers to read.

        A trainer that joins midway cannot infer any of this from the stream.
        ``dataset_size`` in particular is not in the config -- it is whatever
        survives loading and filtering, so only the producer knows it, and only
        at runtime.

        Must be called BEFORE the first XADD, or a fast consumer could read a
        sample and find no plan. Raises on failure: without this a consumer
        cannot size its LR schedule, so failing at startup beats failing subtly
        an hour in.

        A late joiner cannot see samples published before it arrived, so it
        cannot do "N epochs" in the strict sense. What it can do is size its
        schedule from what remains::

            remaining = total_samples_planned - producer_idx_at_join
            steps     = remaining / its own global_batch_size

        That biases high, since drops reduce the real count -- the safe
        direction, because EOS stops the run regardless.
        """
        num_epochs = getattr(self.args, "num_epochs", 1)
        meta = {
            "v": STREAM_SCHEMA_VERSION,
            # Guards against a consumer reading a previous run's plan if this
            # ever points at a Redis that outlives the job.
            "run_id": str(os.environ.get("JOB_ID", "unknown")),
            "status": RUN_STATUS_RUNNING,
            "started_at": str(time.time()),
            "stream": self.redis_train_stream,
            # --- the run plan ---
            "dataset_size": str(dataset_size),
            "num_epochs": str(num_epochs),
            "total_samples_planned": str(dataset_size * num_epochs),
            # --- the retention contract ---
            "eviction_ttl_seconds": str(self._eviction_ttl),
            # --- shapes, so a consumer can assert its draft config matches
            # before building the model rather than hitting a reshape error
            # deep in the Mooncake fetch path ---
            "hidden_dim": str(getattr(self.args, "mooncake_hidden_dim", "")),
            "num_aux_layers": str(getattr(self.args, "mooncake_num_aux_layers", "")),
            "max_seq_len": str(getattr(self.args, "mooncake_max_seq_len", "")),
        }
        self._redis_client.hset(self._meta_key, mapping=meta)
        logger.info(
            "Published run meta to %s: dataset_size=%s num_epochs=%s "
            "total_samples_planned=%s ttl=%ss",
            self._meta_key, dataset_size, num_epochs,
            meta["total_samples_planned"], self._eviction_ttl,
        )
        return meta

    def _set_run_status(self, status: str) -> None:
        """Mark the run finished. Best-effort: EOS is the primary signal."""
        try:
            self._redis_client.hset(self._meta_key, mapping={
                "status": status, "ended_at": str(time.time()),
                "published": str(self._producer_idx),
                "dropped": str(self._publish_dropped),
            })
        except Exception:
            logger.warning("Failed to set run status=%s on %s", status, self._meta_key,
                           exc_info=True)

    def set_mooncake_store(self, mooncake_store):
        self._mooncake_store = mooncake_store

    def _generate_data_id(self) -> str:
        self._data_id_counter += 1
        return f"data_{self._data_id_counter}"

    # ─────────────────────────────────────────────────────────────
    # Dataset Loading
    # ─────────────────────────────────────────────────────────────

    def add_dataset(self, dataset: list) -> int:
        with self._prompt_lock:
            for sample in dataset:
                if isinstance(sample, dict):
                    data_id = sample.get("data_id") or self._generate_data_id()
                    input_ids = sample.get("input_ids")
                    packed_loss_mask = sample.get("packed_loss_mask")
                    if (
                        input_ids is not None
                        and packed_loss_mask is None
                        and not getattr(self.args, "dynamic_loss_mask", False)
                    ):
                        raise ValueError(
                            f"packed_loss_mask is required when input_ids is provided "
                            f"(data_id={data_id}). Enable dynamic_loss_mask or use "
                            f"defer_tokenization=True to match engine-produced token IDs."
                        )
                    entry = InferenceInput(
                        data_id=data_id,
                        prompt=sample.get("prompt", sample),
                        input_ids=input_ids,
                        packed_loss_mask=packed_loss_mask,
                        formatted_prompt=sample.get("formatted_prompt"),
                        metadata=sample.get("metadata", {}),
                        multimodal_inputs=sample.get("multimodal_inputs"),
                    )
                else:
                    entry = InferenceInput(
                        data_id=self._generate_data_id(),
                        prompt=sample,
                    )
                self.prompt_buffer.append(entry)
            return len(dataset)

    def _load_dataset_split(self, args, split: str) -> list:
        """Load one split from either replay records or conversation data."""
        if split not in ("train", "eval"):
            raise ValueError(f"Unknown dataset split: {split!r}")

        if getattr(args, "inference_engine_type", None) == "offline":
            from torchspec.offline.dataset import OfflineDataset

            dataset = OfflineDataset(args.offline_data_path)
            return [
                {
                    "data_id": str(row["data_id"]),
                    "metadata": {"offline_replay": True},
                    "seq_len": row["seq_len"],
                }
                for row in dataset.rows(split)
            ]

        data_path = (
            args.train_data_path if split == "train" else getattr(args, "eval_data_path", None)
        )
        if not data_path:
            return []

        from torchspec.data.dataset import load_conversation_dataset

        dataset_args = args
        if split == "eval":
            dataset_args = copy.copy(args)
            dataset_args.train_data_path = data_path
            if getattr(args, "eval_prompt_key", None):
                dataset_args.prompt_key = args.eval_prompt_key
        return load_conversation_dataset(dataset_args)

    def load_dataset(self, args) -> int:
        """Load and store the training dataset for later epochs."""
        self._stored_dataset = self._load_dataset_split(args, "train")
        if not self._stored_dataset:
            if getattr(args, "inference_engine_type", None) == "offline":
                raise ValueError("Offline dataset has no train samples")
            raise ValueError(
                f"Training dataset is empty after processing. "
                f"Check train_data_path='{args.train_data_path}', "
                f"max_seq_length={getattr(args, 'max_seq_length', None)}, "
                f"and chat_template settings."
            )
        logger.info(f"Controller loaded dataset: {len(self._stored_dataset)} samples")
        return len(self._stored_dataset)

    def _prepare_dataset(self, skip: int = 0) -> list:
        """Return dataset for the current epoch, optionally shuffled and length-grouped.

        When shuffle is enabled the ordering is deterministic from
        (seed + epoch), so resume can reconstruct the same epoch ordering
        and approximately skip samples consumed by completed optimizer
        steps.  This is best-effort only because async prompt/result
        buffers may still contain in-flight samples.
        """
        data = list(self._stored_dataset)
        if self._shuffle_dataset:
            import random

            rng = random.Random(self._dataset_seed + self._dataset_epoch)
            rng.shuffle(data)

        data = length_grouped_order(data, self._length_group_size)

        if skip > 0:
            skip = min(skip, len(data))
            data = data[skip:]

        shuffle_tag = (
            f"seed {self._dataset_seed}+{self._dataset_epoch}"
            if self._shuffle_dataset
            else "shuffle disabled"
        )
        logger.info(
            f"Prepared dataset ({len(data)} samples, {shuffle_tag}, "
            f"length group {self._length_group_size}" + (f", skipped {skip})" if skip > 0 else ")")
        )
        return data

    def submit_training_dataset(self, epoch: int = 0, skip: int = 0) -> int:
        """Submit the stored training dataset to the prompt buffer for inference.

        Args:
            epoch: Current epoch number (for deterministic shuffle seed).
            skip: Number of samples to skip from the start (for resume mid-epoch).
        """
        assert self._stored_dataset is not None, "No stored dataset to submit"
        self._dataset_epoch = epoch
        return self.add_dataset(self._prepare_dataset(skip=skip))

    def reload_dataset(self) -> int:
        """Re-add the stored dataset to the prompt buffer (epoch reload)."""
        assert self._stored_dataset is not None, "No stored dataset to reload"
        self._dataset_epoch += 1
        return self.add_dataset(self._prepare_dataset())

    def load_eval_dataset(self, args) -> int:
        """Not supported: the eval path still assumes per-DP-rank queues."""
        raise NotImplementedError(
            "Eval is not supported by the Redis-based controller. try_dispatch_eval_batch "
            "and finalize_eval_dispatch still need a batch size that used to come from "
            "dp_size, which now lives on the consumer side."
        )

    def get_dataset_size(self) -> int:
        if self._stored_dataset is None:
            raise RuntimeError(
                "get_dataset_size() called but no dataset has been loaded. "
                "Call load_dataset() first."
            )
        return len(self._stored_dataset)

    def get_eval_dataset_size(self) -> int:
        return len(self._stored_eval_dataset) if self._stored_eval_dataset is not None else 0

    def compute_vocab_mapping(self, target_vocab_size: int, draft_vocab_size: int) -> tuple:
        """Generate vocab mapping on the controller using the stored dataset.

        Requires the dataset to have been loaded with defer_tokenization=False,
        since vocab mapping needs input_ids.
        """
        from torchspec.data.preprocessing import generate_vocab_mapping

        assert self._stored_dataset is not None, "No stored dataset for vocab mapping"
        assert "input_ids" in self._stored_dataset[0], (
            "compute_vocab_mapping requires input_ids in dataset. "
            "Set defer_tokenization=False to enable tokenization."
        )
        return generate_vocab_mapping(
            prompts=self._stored_dataset,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

    # ─────────────────────────────────────────────────────────────
    # Interface for Inference Manager
    # ─────────────────────────────────────────────────────────────

    def get_prompts(self, num_prompts: int) -> list[InferenceInput]:
        """Inference manager gets prompts with data_ids.

        Args:
            num_prompts: Maximum number of prompts to fetch.

        Returns:
            List of InferenceInput objects.
        """
        with self._prompt_lock:
            entries = []
            for _ in range(min(num_prompts, len(self.prompt_buffer))):
                entries.append(self.prompt_buffer.popleft())
            return entries

    def push_inference_results(self, results: list[InferenceOutput]) -> int:
        """Inference sends back (data_id, mooncake_key) pairs.

        Controller stores the keys and tracks exact bytes for backpressure.
        Eval results (identified by data_id) are routed to the eval pool.

        Args:
            results: List of InferenceOutput containing data_id, mooncake_key,
                    tensor_shapes, and tensor_dtypes.

        Returns:
            Current pool bytes after adding results. This allows inference manager
            to implement Mooncake backpressure.
        """
        eval_results = []
        train_results = []
        for result in results:
            if result.data_id in self._eval_data_ids:
                eval_results.append(result)
            else:
                train_results.append(result)

        if eval_results:
            with self._eval_pool_lock:
                self.eval_pool.extend(eval_results)

        pool_bytes = 0
        if train_results:
            if not self.verified_tensor_shapes:
                self.verified_tensor_shapes = True
                try:
                    self._verify_tensor_shapes(train_results[0])
                except Exception:
                    logger.exception("Round-trip verification failed for %s", train_results[0].mooncake_key)
                    self.set_inference_error("mooncake round-trip verification failed")

            with self._pool_lock:
                for result in train_results:
                    sample_bytes = estimate_tensor_bytes(
                        result.tensor_shapes or {},
                        result.tensor_dtypes or {},
                    )
                    self._sample_bytes[result.mooncake_key] = sample_bytes
                    self._pool_bytes += sample_bytes
                    shapes = result.tensor_shapes or {}
                    mooncake_entry = MooncakeEntry(
                            inserted_at=time.time(),
                            dispatched_at=None,
                            num_bytes=sample_bytes,
                            has_last_hidden_states="last_hidden_states" in shapes,
                            has_target="target" in shapes,
                    )
                    self._mooncake_entries[result.mooncake_key] = mooncake_entry
                    self._mooncake_bytes += sample_bytes
                self.sample_pool.extend(train_results)
                pool_bytes = self._pool_bytes

        self._inference_monitor.record(len(results))
        return pool_bytes

    def get_prompt_buffer_size(self) -> int:
        """Get current size of prompt buffer."""
        return len(self.prompt_buffer)

    # ─────────────────────────────────────────────────────────────
    # Interface for Training
    # ─────────────────────────────────────────────────────────────

    def get_pool_size(self) -> int:
        """Total mooncake-resident samples (training + eval) for backpressure.

        Always includes eval pool so that backpressure accounts for mooncake
        segment capacity used by eval data.  Without this, eval data occupies
        mooncake outside backpressure awareness and the segment overflows.
        """
        train_size = len(self.sample_pool)
        with self._eval_pool_lock:
            return train_size + len(self.eval_pool)

    def get_pool_bytes(self) -> int:
        """Get current bytes in sample pool (for Mooncake backpressure)."""
        with self._pool_lock:
            return self._pool_bytes

    def get_mooncake_bytes(self) -> int:
        """Get current bytes in."""
        with self._pool_lock:
            return self._mooncake_bytes

    # ─────────────────────────────────────────────────────────────
    # Dispatch Logic
    # ─────────────────────────────────────────────────────────────

    def set_inference_error(self, msg: str) -> None:
        with self._error_lock:
            self._inference_error = msg
            self._consecutive_errors += 1

    def clear_inference_error(self) -> None:
        with self._error_lock:
            self._inference_error = None
            self._consecutive_errors = 0

    def try_dispatch_batch(self) -> bool:
        """Try to dispatch one batch to Redis.

        Only dispatches when sample pool has enough samples (>= dispatch_batch_size).
        Dispatches TrainSample objects that MooncakeDataFetcher can consume.
        Subtracts bytes from pool tracking when dispatching.

        Returns:
            True if a batch was dispatched, False if not enough samples.

        Raises:
            RuntimeError: If the inference manager has reported a fatal error.
        """
        with self._error_lock:
            if self._inference_error is not None:
                if self._consecutive_errors >= 10:
                    logger.error(
                        f"Too many consecutive inference manager failures: {self._inference_error}"
                    )
                    raise RuntimeError(f"Inference engine failed: {self._inference_error}")
                # The caller polls this at ~20Hz, so log at most once every 2s
                # (matching the pool-size log below) instead of once per poll.
                now = time.time()
                if now - self._last_error_log_time >= 2.0:
                    self._last_error_log_time = now
                    logger.error(
                        f"Inference manager failed ({self._consecutive_errors}/10): "
                        f"{self._inference_error}"
                    )
            if self._publish_failures >= 10:
                raise RuntimeError(f"Redis publish failed {self._publish_failures} times consecutively")

        with self._pool_lock:
            pool_size = len(self.sample_pool)
            now = time.time()
            should_log = (now - self._last_dispatch_log_time) >= 2.0
            if pool_size < self.dispatch_batch_size:
                if should_log:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"try_dispatch_batch: pool_size={pool_size} < dispatch_batch_size={self.dispatch_batch_size}, not dispatching"
                    )
                return False

            if should_log:
                self._last_dispatch_log_time = now
                logger.debug(
                    f"try_dispatch_batch: pool_size={pool_size} >= dispatch_batch_size={self.dispatch_batch_size}, dispatching batch {self.batch_id}"
                )

            batch_results = []
            for _ in range(self.dispatch_batch_size):
                result = self.sample_pool.popleft()
                sample_bytes = self._sample_bytes.pop(result.mooncake_key, 0)
                self._pool_bytes -= sample_bytes
                entry = self._mooncake_entries.get(result.mooncake_key)
                if entry is not None:
                    entry.dispatched_at = time.time()      # TTL runs from dispatch, not arrival
                batch_results.append(result)

        number_dispatched = self._dispatch_to_redis(batch_results, self.redis_train_stream)

        self._training_monitor.record(number_dispatched)
        logger.debug(
            f"Attempted to dispatch batch {self.batch_id} with {self.dispatch_batch_size} samples "
            f"to Redis stream at t={time.time():.3f} "
            f"Actual number dispatched: {number_dispatched} "
        )
        self.batch_id += 1
        return True

    def _dispatch_to_redis(self, batch_results: list[InferenceOutput], stream_name: str) -> int:
        """Publish to the Redis stream; returns the number actually published.

        try_dispatch_batch has already popped these from sample_pool and started
        their eviction clock, so a failure here loses them for good. Retry inside
        the TTL rather than rolling back, and count failures rather than raising —
        raising propagates into training_loop and kills the run on a blip.
        """
        samples = []
        for result in batch_results:
            entry = self._mooncake_entries.get(result.mooncake_key)
            if entry is None or entry.dispatched_at is None:
                # No retention guarantee we can state. Publishing anyway would send
                # expires_at absent -> consumer reads 0.0 -> treats it as expired.
                logger.error("No mooncake entry for %s; not publishing", result.mooncake_key)
                self._publish_dropped += 1
                continue
            metadata = getattr(result, "metadata", {}) or {}
            samples.append(TrainSampleRedis(
                mooncake_key=result.mooncake_key,
                tensor_shapes=result.tensor_shapes,
                tensor_dtypes=result.tensor_dtypes,
                packed_loss_mask=result.packed_loss_mask,
                expires_at=entry.dispatched_at + self._eviction_ttl,
                producer_idx=self._producer_idx,
                last_turn_loss_only=metadata.get("has_thinking"),
                metadata=metadata,
                data_id=result.data_id,
            ))
            self._producer_idx += 1

        pending = samples
        for attempt in range(1, self._publish_max_attempts + 1):
            pending = self._xadd_batch(pending, stream_name)
            if not pending:
                break
            if attempt < self._publish_max_attempts:
                time.sleep(self._publish_retry_seconds)

        if pending:
            self._publish_dropped += len(pending)
            self._publish_failures += 1
            logger.error("Dropped %d samples after %d publish attempts (%d consecutive)",
                         len(pending), self._publish_max_attempts, self._publish_failures)
        elif samples:
            self._publish_failures = 0

        return len(samples) - len(pending)


    def _xadd_batch(self, samples: list[TrainSampleRedis], stream_name: str) -> list[TrainSampleRedis]:
        """One pipelined round trip. Returns only the samples that failed."""
        if not samples:
            return []
        pipe = self._redis_client.pipeline(transaction=False)
        for s in samples:
            pipe.xadd(
                name=stream_name,
                fields=s.to_redis_fields(),
                id="*",                       # Redis assigns <ms>-<seq>
                maxlen=self._stream_maxlen,
                approximate=True,             # MAXLEN ~ N: trims whole nodes, O(1)
            )
        try:
            results = pipe.execute(raise_on_error=False)
        except redis.RedisError as exc:
            logger.warning("Redis pipeline failed entirely: %s", exc)
            return samples
        return [s for s, r in zip(samples, results) if isinstance(r, Exception)]


    def push_inference_sample(self, sample: InferenceOutput) -> int:
        """Add a single inference sample to the training pool.

        This method is used by OnlineServingController to add samples
        captured from the inference engine's /generate API.

        Args:
            sample: InferenceOutput containing mooncake_key and tensor metadata.

        Returns:
            Current pool bytes after adding the sample.
        """
        return self.push_inference_results([sample])


    def _verify_tensor_shapes(self, result: InferenceOutput):
        shapes = result.tensor_shapes or {}
        mooncake_key = result.mooncake_key
        dtypes = {k: (getattr(torch, v.replace("torch.", "")) if isinstance(v, str) else v) for k, v in (result.tensor_dtypes or {}).items()}
        logger.info("SAMPLE hs=%s lhs=%s", shapes.get("hidden_states"), shapes.get("last_hidden_states"))
        out = self._mooncake_store.get(mooncake_key, shapes, dtypes, torch.device("cpu"))
        logger.info("ROUNDTRIP hs=%s lhs=%s", out.hidden_states.shape, out.last_hidden_states.shape)


    def _eviction_sweep(self) -> None:
        """One eviction pass: drop expired entries, then delete them from Mooncake.

        An entry is expired once it has left the sample pool (so a trainer has
        actually been handed the key) and has been dispatched for longer than
        ``_eviction_ttl``.
        """
        now = time.time()
        with self._pool_lock:
            expired = [
                (k, e) for k, e in self._mooncake_entries.items()
                if k not in self._sample_bytes
                and e.dispatched_at is not None
                and now - e.dispatched_at >= self._eviction_ttl
            ]
            for k, e in expired:
                del self._mooncake_entries[k]
                self._mooncake_bytes -= e.num_bytes
        if expired:
            logger.info("Evicted %d entries, %.2f GiB resident", len(expired), self._mooncake_bytes / 1024**3)
        # Do the actual store removal *outside* the lock — it can block on
        # Mooncake and we don't want to stall push_inference_results.
        for k, e in expired:
            try:
                self._mooncake_store.remove_eagle3_tensors(
                    k,
                    has_last_hidden_states=e.has_last_hidden_states,
                    has_target=e.has_target,
                    raise_on_failure=True
                )
            except Exception:
                logger.exception("Eviction sweep failed for %s", k)
                # If deletion fails, restore the entry to the dict so
                # that it will be retried on the next eviction sweep.
                with self._pool_lock:
                    self._mooncake_entries[k] = e
                    self._mooncake_bytes += e.num_bytes
        # Time-align the stream with the store. Guarded separately from the
        # Mooncake removals above: a Redis fault here would otherwise surface as
        # "Eviction sweep raised", blaming eviction for a publishing problem.
        # The 2x keeps the stream strictly outliving the store -- an entry whose
        # tensors are gone is detectable via expires_at, a trimmed one is not.
        cutoff_ms = int((time.time() - 2 * self._eviction_ttl) * 1000)
        for stream in (self.redis_train_stream, self.redis_eval_stream):
            try:
                self._redis_client.xtrim(stream, minid=cutoff_ms, approximate=True)
            except Exception:
                logger.warning("XTRIM failed for stream %s", stream, exc_info=True)

        self._maybe_heartbeat()

    def _maybe_heartbeat(self) -> None:
        """Emit a liveness entry so consumers can distinguish slow from dead.

        Best-effort: a failed heartbeat must not abort the eviction sweep, and
        it deliberately does not touch _publish_failures -- losing a ping is
        not the same as losing data.
        """
        now = time.time()
        if now - self._last_heartbeat < self._heartbeat_seconds:
            return
        self._last_heartbeat = now
        try:
            self._redis_client.xadd(
                name=self.redis_train_stream,
                fields=heartbeat_fields(self._producer_idx, self._dataset_epoch),
                id="*",
                maxlen=self._stream_maxlen,
                approximate=True,
            )
        except Exception:
            logger.debug("Heartbeat XADD failed", exc_info=True)

    def _eviction_loop(self):
        """Sweep until stopped, surviving any failure in a single sweep.

        This runs on a daemon thread, so an escaping exception would kill it
        silently and stop eviction for the rest of the run — which surfaces
        much later as a Mooncake watermark that never recedes and generation
        paused indefinitely. Failures are throttled rather than logged every
        interval, since a persistent fault would otherwise flood the log.
        """
        consecutive_failures = 0
        last_failure_log = 0.0
        while not self._eviction_stop.wait(self._eviction_interval):
            try:
                self._eviction_sweep()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                now = time.monotonic()
                if now - last_failure_log >= 30.0:
                    last_failure_log = now
                    logger.exception(
                        "Eviction sweep raised (%d consecutive failures); "
                        "retrying every %.1fs",
                        consecutive_failures,
                        self._eviction_interval,
                    )

    def start_eviction_sweeper(self):
        if self._mooncake_store is None:
            raise RuntimeError("mooncake_store required for time-based eviction")
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, name="controller-evictor", daemon=True,
        )
        self._eviction_thread.start()

    # ─────────────────────────────────────────────────────────────
    # Eval Pipeline
    # ─────────────────────────────────────────────────────────────

    def _build_eval_entries(self, dataset: list) -> list[InferenceInput]:
        eval_entries: list[InferenceInput] = []
        for sample in dataset:
            if isinstance(sample, dict):
                raw_id = sample.get("data_id") or self._generate_data_id()
                data_id = (
                    str(raw_id)
                    if getattr(self.args, "inference_engine_type", None) == "offline"
                    else f"eval_{raw_id}"
                )
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(
                    data_id=data_id,
                    prompt=sample.get("prompt", sample),
                    input_ids=sample.get("input_ids"),
                    packed_loss_mask=sample.get("packed_loss_mask"),
                    formatted_prompt=sample.get("formatted_prompt"),
                    metadata=sample.get("metadata", {}),
                    multimodal_inputs=sample.get("multimodal_inputs"),
                )
            else:
                data_id = f"eval_{self._generate_data_id()}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(data_id=data_id, prompt=sample)
            eval_entries.append(entry)
        return eval_entries

    def submit_eval_chunk(self, start: int, end: int) -> int:
        """Submit a slice of the stored eval dataset for inference."""
        assert self._stored_eval_dataset is not None, "No stored eval dataset"
        chunk = self._stored_eval_dataset[start:end]
        if not chunk:
            return 0

        if start == 0:
            self._eval_expected_count = len(self._stored_eval_dataset)
            self._eval_dispatched_samples = 0

        eval_entries = self._build_eval_entries(chunk)

        with self._prompt_lock:
            self.prompt_buffer.extendleft(reversed(eval_entries))
        logger.info(
            f"Eval: submitted chunk [{start}:{end}] "
            f"({len(chunk)} samples, total_expected={self._eval_expected_count})"
        )
        return len(chunk)

    def get_eval_pool_size(self) -> int:
        with self._eval_pool_lock:
            return len(self.eval_pool)

    def try_dispatch_eval_batch(self) -> bool:
        """Dispatch one eval batch from the pool if enough samples are available."""
        if self.eval_dispatch_batch_size is None:
            raise NotImplementedError("Eval dispatch requires a batch size; see load_eval_dataset")
        bs = self.eval_dispatch_batch_size
        with self._eval_pool_lock:
            if len(self.eval_pool) < bs:
                return False
            batch_results = [self.eval_pool.popleft() for _ in range(bs)]

        self._dispatch_to_redis(batch_results, self.redis_eval_stream)
        self._eval_dispatched_samples += bs
        logger.debug(
            f"Eval: dispatched batch ({self._eval_dispatched_samples}/"
            f"{self._eval_expected_count} samples)"
        )
        return True

    def finalize_eval_dispatch(self) -> None:
        """Assert all eval batches were dispatched, then clean up tracking state.

        Raises AssertionError if not all expected samples have arrived or
        undispatched full batches remain in the pool.
        """
        if self.eval_dispatch_batch_size is None:
            raise NotImplementedError("Eval dispatch requires a batch size; see load_eval_dataset")
        with self._eval_pool_lock:
            arrived = self._eval_dispatched_samples + len(self.eval_pool)
            pool_remaining = len(self.eval_pool)

        assert self._eval_expected_count > 0 and arrived >= self._eval_expected_count, (
            f"finalize_eval_dispatch called before all samples arrived "
            f"(arrived={arrived}, expected={self._eval_expected_count})"
        )
        assert pool_remaining < self.eval_dispatch_batch_size, (
            f"finalize_eval_dispatch called with undispatched full batches "
            f"(pool={pool_remaining}, batch_size={self.eval_dispatch_batch_size})"
        )

        with self._eval_pool_lock:
            if pool_remaining > 0:
                logger.info(
                    f"Eval: dropping {pool_remaining} leftover samples that didn't fill a batch"
                )
                self.eval_pool.clear()

        self._eval_data_ids.clear()
        self._eval_expected_count = 0
        self._eval_dispatched_samples = 0
        logger.info("Eval: dispatch finalized, tracking state cleared")

    # ─────────────────────────────────────────────────────────────
    # Status and Monitoring
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current status of controller."""
        return {
            "prompt_buffer_size": len(self.prompt_buffer),
            "sample_pool_size": len(self.sample_pool),
            "batches_dispatched": self.batch_id,
            "published": self._producer_idx,
            "publish_dropped": self._publish_dropped,
            "dispatch_batch_size": self.dispatch_batch_size,
        }

    def get_speeds(self) -> dict:
        """Get current throughput speeds in entries/sec."""
        elapsed = time.time() - self._start_time
        return {
            "inference_speed": round(self._inference_monitor.get_speed(), 2),
            "training_speed": round(self._training_monitor.get_speed(), 2),
            "inference_total": self._inference_monitor.get_total_count(),
            "training_total": self._training_monitor.get_total_count(),
            "elapsed_seconds": round(elapsed, 1),
            "avg_inference_speed": round(
                self._inference_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
            "avg_training_speed": round(
                self._training_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
        }

    def get_full_status(self) -> dict:
        """Get complete status including speeds."""
        status = self.get_status()
        status.update(self.get_speeds())
        return status

    def drain_pool(self) -> list[InferenceOutput]:
        with self._pool_lock:
            leftovers = list(self.sample_pool)
            self.sample_pool.clear()
            for r in leftovers:
                self._pool_bytes -= self._sample_bytes.pop(r.mooncake_key, 0)
        return leftovers

    def shutdown(self) -> None:
        """Signal training workers to stop by sending None to queues."""
        self._eviction_stop.set()
        if self._eviction_thread:
            self._eviction_thread.join(timeout=10)
        # Final sweep regardless of age
        with self._pool_lock:
            stragglers = list(self._mooncake_entries.items())
            self._mooncake_entries.clear()
            self._mooncake_bytes = 0
        for k, e in stragglers:
            try:
                self._mooncake_store.remove_eagle3_tensors(
                    k, has_last_hidden_states=e.has_last_hidden_states,
                    has_target=e.has_target,
                    raise_on_failure=True)
            except Exception:
                logger.exception("Final eviction failed for %s", k)
        self._set_run_status(RUN_STATUS_FINISHED)
        try:
            self._redis_client.xadd(name=self.redis_train_stream, fields=eos_fields(), id="*")
        except Exception:
            logger.warning("Failed to publish EOS; consumers will rely on the heartbeat "
                           "going stale", exc_info=True)
        # item 3: NOACK makes loss silent by design, so this counter is the only
        # record that it happened. A sweep whose arms differ here is confounded.
        logger.info(
            "Controller shutdown: published=%d dropped=%d (stream=%s)",
            self._producer_idx, self._publish_dropped, self.redis_train_stream,
        )


