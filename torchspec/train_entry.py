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

"""Training entry point for Eagle3 speculative decoding."""

import argparse
import dataclasses
import os
import sys
import time
import threading
import asyncio

# Fix PyTorch 2.9+ TorchInductor GEMM backend regression: without this,
# FlexAttention backward pass hits NoValidChoicesError and training is 3x slower.
# See Phase E in docs/inference/dflash/training_results.md.
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON")
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Generator

import ray
import torch
from omegaconf import OmegaConf

from torchspec import AutoDraftModelConfig
from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.config.utils import generate_draft_model_config
from torchspec.controller import (
    AsyncTrainingController,
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
)
from torchspec.controller.inference_manager import AsyncInferenceManager
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.transfer.mooncake.utils import launch_mooncake_master, check_mooncake_master_available
from torchspec.utils.logging import init_tracking, logger

_Phase = namedtuple("_Phase", ["name", "duration", "is_async", "blocked"])


class _InitTimer:
    """Lightweight segmented timer for initialization phases."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._phases: list[_Phase] = []
        self._pending: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time a synchronous phase."""
        start = time.time()
        yield
        self._phases.append(_Phase(name, time.time() - start, is_async=False, blocked=0.0))

    def begin_async(self, name: str) -> None:
        """Mark the start of an async operation (e.g., ray.remote dispatch)."""
        self._pending[name] = time.time()

    def wait(self, name: str, refs) -> Any:
        """Wrap ray.get for async phases. Returns the result."""
        if name not in self._pending:
            raise ValueError(f"No async phase '{name}' was started via begin_async()")
        t_before = time.time()
        result = ray.get(refs)
        t_after = time.time()
        dispatch_time = self._pending.pop(name)
        total = t_after - dispatch_time
        blocked = t_after - t_before
        self._phases.append(_Phase(name, total, is_async=True, blocked=blocked))
        return result

    def log_summary(self) -> None:
        total = time.time() - self._t0
        lines = ["Initialization timing:"]
        for p in self._phases:
            suffix = f"  (blocked {p.blocked:.2f}s)" if p.is_async else ""
            lines.append(f"  {p.name:<48s} {p.duration:>8.2f}s{suffix}")
        lines.append(f"  {'─' * 57}")
        lines.append(f"  {'Total':<48s} {total:>8.2f}s")
        logger.info("\n".join(lines))


def parse_config():
    """Parse YAML config and convert to flat args.

    Supports configs with sections matching the Config dataclass:
    model, dataset, training, debug, inference, logging, mooncake, decode.

    The config is flattened via config_to_flat_args(), with prefixed sections:
    mooncake_*, sglang_*, offline_*, decode_*.
    """

    parser = argparse.ArgumentParser(description="Eagle3 speculative decoding training")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--print-config-only", action="store_true", help="Print resolved config and exit"
    )

    args, unknown = parser.parse_known_args()

    config = load_config(
        config_path=args.config, cli_args=unknown if unknown else None, save_snapshot=True
    )

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))

    if args.print_config_only:
        sys.exit(0)

    flat_args = config_to_flat_args(config)

    flat_args.rank = 0
    flat_args.world_size = flat_args.training_num_nodes * flat_args.training_num_gpus_per_node

    defaults = {
        "colocate": False,
        "dp_size": None,
        "save_debug_train_data": None,
    }
    for key, value in defaults.items():
        if not hasattr(flat_args, key) or getattr(flat_args, key) is None:
            setattr(flat_args, key, value)

    _resolve_batch_size(flat_args)
    _validate_usp_args(flat_args)

    if (
        getattr(flat_args, "inference_engine_type", None) == "offline"
        and getattr(flat_args, "max_sample_pool_size", 0) <= 0
    ):
        flat_args.max_sample_pool_size = max(
            flat_args.global_batch_size,
            getattr(flat_args, "inference_batch_size", 1)
            * getattr(flat_args, "offline_num_engines", 1)
            * 2,
        )
        logger.info(
            "Offline replay set max_sample_pool_size=%d for bounded Mooncake staging",
            flat_args.max_sample_pool_size,
        )

    return flat_args


def _maybe_create_scratch_draft(args, train_group):
    """Auto-create scratch draft checkpoint for inference engine if not provided."""
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "decode_speculative_algorithm", None)
        and getattr(args, "decode_speculative_draft_model_path", None) is None
    ):
        scratch_dir = os.path.join(getattr(args, "output_dir", "./outputs"), "scratch_draft_model")
        os.makedirs(scratch_dir, exist_ok=True)
        logger.info(f"Auto-creating scratch draft checkpoint at {scratch_dir}")
        train_group.save_draft_model_for_serving(scratch_dir)
        args.decode_speculative_draft_model_path = scratch_dir
        logger.info(f"Set decode_speculative_draft_model_path = {scratch_dir}")


def _resolve_batch_size(args):
    """Derive dp_size, per_dp_rank_batch_size, dispatch_batch_size, and global_batch_size."""
    world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
        if sp_size <= 0:
            raise ValueError(f"USP requires positive sp_size, got {sp_size}")
        if world_size % sp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by USP sp_size ({sp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // sp_size)
        if dp_size * sp_size != world_size:
            raise ValueError(
                f"dp_size ({dp_size}) * sp_size ({sp_size}) must equal world_size ({world_size})"
            )
        args.dp_size = dp_size
        args.sp_size = sp_size
        args.per_dp_rank_batch_size = 1
    else:
        dp_size = getattr(args, "dp_size", None) or world_size
        args.dp_size = dp_size
        sp_size = getattr(args, "sp_size", None)
        if sp_size is not None and sp_size != 1:
            raise NotImplementedError(
                f"Sequence parallel is not yet supported (got sp_size={sp_size})"
            )
        sp_size = sp_size or 1
        args.per_dp_rank_batch_size = args.micro_batch_size * sp_size

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps


def _validate_usp_args(args) -> None:
    if getattr(args, "attention_backend", None) != "usp":
        return

    sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
    if sp_size <= 1:
        raise NotImplementedError(f"USP requires sp_size > 1, got {sp_size}")

    inference_engine_type = getattr(args, "inference_engine_type", "sgl")
    if inference_engine_type != "sgl":
        raise ValueError(
            f"USP currently only supports inference_engine_type=sgl, got {inference_engine_type}"
        )

    fsdp_strategy = getattr(args, "fsdp_strategy", "REPLICATE").upper()
    if fsdp_strategy != "REPLICATE":
        raise NotImplementedError(
            f"USP currently only supports fsdp_strategy=REPLICATE, got {fsdp_strategy}"
        )

    micro_batch_size = getattr(args, "micro_batch_size", 1)
    if micro_batch_size != 1:
        raise NotImplementedError(
            f"USP currently only supports micro_batch_size=1, got {micro_batch_size}"
        )


def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        return AutoDraftModelConfig.from_file(draft_config_path)

    config_dict = generate_draft_model_config(
        target_model_path=args.target_model_path,
        cache_dir=getattr(args, "model_download_dir", None),
    )
    return AutoDraftModelConfig.from_dict(config_dict)


def _validate_and_configure_dflash(args, draft_model_config) -> None:
    """Validate DFlash-specific config and auto-set aux layer IDs.

    Called before dataset loading to fail fast on misconfigurations.
    """
    from torchspec.models.draft.dflash import DFlashConfig
    from torchspec.models.draft.dflash2 import DFlash2Config
    from torchspec.models.draft.dspark import DSparkConfig

    if not isinstance(draft_model_config, DFlashConfig):
        return

    is_dflash2 = isinstance(draft_model_config, DFlash2Config)
    is_dspark = isinstance(draft_model_config, DSparkConfig)
    algo = "DFlash2" if is_dflash2 else "DSpark" if is_dspark else "DFlash"

    if is_dflash2 and getattr(args, "attention_backend", None) == "usp":
        raise ValueError("DFlash2 does not support training.attention_backend=usp.")

    engine_type = getattr(args, "inference_engine_type", "hf")
    if engine_type not in ("vllm", "sgl", "trtllm", "offline"):
        raise NotImplementedError(
            f"{algo} supports inference_engine_type in "
            f"('vllm', 'sgl', 'trtllm', 'offline'), got '{engine_type}'."
        )
    if getattr(args, "defer_tokenization", False):
        raise NotImplementedError(f"{algo} does not support defer_tokenization=True.")
    block_size = getattr(args, "dflash_block_size", 16)
    if is_dflash2 and block_size != draft_model_config.block_size:
        raise ValueError(
            "training.dflash_block_size must match dflash_config.block_size "
            f"({block_size} != {draft_model_config.block_size})."
        )
    num_target_layers = getattr(args, "dflash_num_target_layers", 5)
    if is_dflash2 and num_target_layers != draft_model_config.num_target_layers:
        raise ValueError(
            "training.dflash_num_target_layers must match the number of "
            f"dflash_config.target_layer_ids ({num_target_layers} != "
            f"{draft_model_config.num_target_layers})."
        )
    min_loss = getattr(args, "min_loss_tokens", 0)
    if min_loss < 2 * block_size:
        raise ValueError(
            f"{algo} requires dataset.min_loss_tokens >= 2 * training.dflash_block_size "
            f"({min_loss} < {2 * block_size}). Set dataset.min_loss_tokens={2 * block_size}."
        )

    target_layer_ids = getattr(draft_model_config, "target_layer_ids", None)
    if not getattr(args, "aux_hidden_states_layers", None):
        from torchspec.models.draft.dflash import build_target_layer_ids

        if target_layer_ids is None:
            num_target = getattr(draft_model_config, "num_target_layers", 5)
            target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", 36)
            target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
        args.aux_hidden_states_layers = target_layer_ids
        logger.info(f"{algo}: set aux_hidden_states_layers = {target_layer_ids}")
    elif is_dflash2 and list(args.aux_hidden_states_layers) != list(target_layer_ids):
        raise ValueError(
            "inference.aux_hidden_states_layers must match "
            f"dflash_config.target_layer_ids ({list(args.aux_hidden_states_layers)} != "
            f"{list(target_layer_ids)})."
        )


def train_async_no_generation(args):
    """Entry point for Eagle3 asynchronous training.

    Supports prefill-only mode (default) and decode mode (train_with_decode=True)
    with speculative decoding. Uses distributed Ray actors with placement groups.
    Engines store tensors in mooncake and return keys to AsyncInferenceManager.
    """
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "inference_engine_type", "sgl") != "sgl"
    ):
        raise ValueError("train_with_decode=True requires inference_engine_type=sgl")

    if getattr(args, "inference_engine_type", None) == "offline":
        from torchspec.offline.dataset import (
            OfflineDataset,
            configure_offline_args,
        )

        offline_dataset = OfflineDataset(args.offline_data_path)
        configure_offline_args(offline_dataset, args)

    init_tracking(args)
    timer = _InitTimer()

    # [1] Create controller early (lightweight: only needs args + dp_size)
    with timer.phase("Create controller"):
        controller = AsyncTrainingController(args, args.dp_size)

    # [2] Kick off dataset loading on controller
    # The original design used Ray to locate the controller in a separate process
    # but that is no longer the case here. There may be some performance impact.
    timer.begin_async("Dataset loading")
    dataset_size = controller.load_dataset(args)

    # [3] Wait for dataset sizes (small ints, unlike the old ray.put of the full dataset)
    logger.info(f"Dataset loaded on controller: {dataset_size} train")

    # [4] Continue with initialization sequentially.
    mooncake_master = None
    with timer.phase("Driver-side init"):
        mooncake_config = build_mooncake_config(args)
        check_mooncake_master_available(mooncake_config.master_server_address, mooncake_config.metadata_server)
        mooncake_config_store = dataclasses.replace(
            mooncake_config,
            global_segment_size=0,        # contributes no storage
            async_put_pool_size=0,        # never puts
            local_buffer_size="1GB",   # never gets
            enable_gpu_direct=False,      # explicit; also skips the GPU receive buffer
        )
        mooncake_store = EagleMooncakeStore(mooncake_config_store)
        mooncake_store.setup(device=torch.device("cpu"))


    # [5] Auto-calculate training steps (needs dataset_size)
    with timer.phase("Auto-calculate training steps"):
        auto_calculate_training_steps(args, dataset_size)

    # [9] Setup async training with pre-created controller
    with timer.phase("Setup async training"):
        inference_manager = AsyncInferenceManager(args, controller, mooncake_config)

    mgr_thread = threading.Thread(
        target=lambda: asyncio.run(inference_manager.run()),
        daemon=True, name="inference-manager",
    )
    mgr_thread.start()

    timer.log_summary()

    # [10] Run training loop (no ray.put needed — dataset lives on controller)
    run_training_loop(
        args,
        controller,
        inference_manager,
        mgr_thread,
        mooncake_master,
        mooncake_store,
        dataset_size=dataset_size,
    )


if __name__ == "__main__":
    args = parse_config()
    train_async_no_generation(args)
