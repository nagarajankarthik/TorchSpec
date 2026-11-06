"""Persisted record of which dataset rows have been published, for resume.

A generation job on a walltime-limited cluster gets killed mid-epoch, and the
loop's step count cannot say which rows were done: samples are published in
completion order, not dataset order, and a failed inference request leaves a
hole that is never revisited in that run. So this tracks rows directly -- one
bit per row per epoch, keyed by ``data_id`` -- and a restarted job resubmits
exactly the rows whose bit is still clear, in the epoch's usual order.

"Published" means the XADD to the Redis stream succeeded. Two windows remain:

- A row published after the last save is published again by the next job (a
  duplicate). The loop saves periodically and again on SIGTERM, which is what
  PBS sends at walltime, so in practice this window is close to empty.
- A row published but not yet fetched by a consumer when the job dies is lost:
  its tensors lived in this job's Mooncake segment. Nothing on the producer
  side can see consumer progress, so this is not tracked.

Epoch attribution: the next epoch is submitted while the previous one's tail
may still be in flight, so the same ``data_id`` can be pending in two epochs.
A published row is credited to the oldest open epoch that has not yet seen it.
That can credit the "wrong" copy, but the set of rows still owed is the same
either way, which is all resume needs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
import zlib
from typing import Any, Iterable

from torchspec.utils.logging import logger

FORMAT_VERSION = 1
DEFAULT_FILENAME = "generation_progress.json"


class ProgressMismatchError(ValueError):
    """The progress file was written for a different dataset."""


def resolve_progress_path(args) -> str | None:
    """Return where progress is kept, or None if tracking is off.

    ``training.generation_progress_file`` wins; otherwise the file lives in
    ``output_dir``. With neither set there is nowhere stable to put it.
    """
    path = getattr(args, "generation_progress_file", None)
    if path:
        return os.path.abspath(os.path.expanduser(path))
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        return os.path.join(os.path.abspath(os.path.expanduser(output_dir)), DEFAULT_FILENAME)
    return None


def check_progress_file(path: str | None, resume: bool) -> None:
    """Fail before the (slow) dataset load if the run would clobber progress.

    Starting fresh over an existing file would overwrite the record a chained
    job needs, so that has to be asked for explicitly.
    """
    if path is None:
        logger.warning(
            "Generation progress is not tracked: set output_dir or "
            "training.generation_progress_file to make this run resumable"
        )
        return
    if os.path.exists(path) and not resume:
        raise FileExistsError(
            f"Generation progress file {path} exists from an earlier run. Set "
            f"training.resume_generation=true to continue that run, or move the "
            f"file away to start over."
        )


def _hash_ids(data_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for data_id in data_ids:
        digest.update(data_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


class GenerationProgress:
    """Per-epoch bitmaps of published rows, plus opaque loop state."""

    def __init__(self, data_ids: list[str], *, fingerprint: dict[str, Any] | None = None):
        self._index: dict[str, int] = {}
        for i, data_id in enumerate(data_ids):
            if data_id in self._index:
                raise ValueError(f"Duplicate data_id {data_id!r}; rows cannot be tracked")
            self._index[data_id] = i
        self.size = len(data_ids)
        self.fingerprint = {
            **(fingerprint or {}),
            "dataset_size": self.size,
            "data_ids_sha256": _hash_ids(data_ids),
        }
        # Only epochs with rows still owed are kept; a full epoch is dropped.
        self._published: dict[int, bytearray] = {}
        self._counts: dict[int, int] = {}
        self.last_epoch: int | None = None
        self.loop_state: dict[str, Any] = {}
        self.resumed = False
        self.unknown_ids = 0
        self.unattributed = 0

    # -- tracking ------------------------------------------------------------

    def open_epoch(self, epoch: int) -> None:
        if epoch not in self._published:
            self._published[epoch] = bytearray((self.size + 7) // 8)
            self._counts[epoch] = 0
        if self.last_epoch is None or epoch > self.last_epoch:
            self.last_epoch = epoch

    @property
    def open_epochs(self) -> list[int]:
        return sorted(self._published)

    def published_in(self, epoch: int) -> int:
        """Rows of ``epoch`` published so far (all of them once it has closed)."""
        if epoch in self._counts:
            return self._counts[epoch]
        if self.last_epoch is not None and epoch <= self.last_epoch:
            return self.size
        return 0

    def is_published(self, epoch: int, data_id: str) -> bool:
        bitmap = self._published.get(epoch)
        if bitmap is None:
            # Closed epochs are complete; unopened ones have nothing yet.
            return self.last_epoch is not None and epoch <= self.last_epoch
        byte, bit = divmod(self._index[data_id], 8)
        return bool(bitmap[byte] & (1 << bit))

    def mark_published(self, data_id: str | None) -> None:
        idx = self._index.get(data_id) if data_id is not None else None
        if idx is None:
            self.unknown_ids += 1
            if self.unknown_ids <= 5:
                logger.warning("Published data_id %r is not in the tracked dataset", data_id)
            return
        byte, bit = divmod(idx, 8)
        mask = 1 << bit
        for epoch in sorted(self._published):
            bitmap = self._published[epoch]
            if bitmap[byte] & mask:
                continue
            bitmap[byte] |= mask
            self._counts[epoch] += 1
            if self._counts[epoch] == self.size:
                del self._published[epoch]
                del self._counts[epoch]
                logger.info("Generation progress: every row of epoch %d published", epoch)
            return
        # Every open epoch already has it: a row published twice in one run.
        self.unattributed += 1

    def resume_epochs(self) -> list[int]:
        """Epochs to (re)submit at startup, oldest first.

        Each open epoch has rows still owed. With none open, the next epoch
        starts from scratch.
        """
        if self._published:
            return self.open_epochs
        return [0 if self.last_epoch is None else self.last_epoch + 1]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "last_epoch": self.last_epoch,
            "open_epochs": {
                str(epoch): {
                    "published": self._counts[epoch],
                    "bitmap_zlib_b64": base64.b64encode(
                        zlib.compress(bytes(self._published[epoch]))
                    ).decode("ascii"),
                }
                for epoch in self.open_epochs
            },
            "loop_state": self.loop_state,
            "saved_at": time.time(),
            "job_id": os.environ.get("JOB_ID"),
        }

    def save(self, path: str) -> None:
        """Write atomically: a kill mid-write leaves the previous file intact."""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".generation_progress.", dir=directory)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        if state.get("version") != FORMAT_VERSION:
            raise ProgressMismatchError(
                f"{path} has format version {state.get('version')!r}, expected {FORMAT_VERSION}"
            )
        saved = state.get("fingerprint", {})
        diffs = {
            key: (saved.get(key), value)
            for key, value in self.fingerprint.items()
            if saved.get(key) != value
        }
        if diffs:
            detail = ", ".join(f"{k}: saved={a!r} now={b!r}" for k, (a, b) in diffs.items())
            raise ProgressMismatchError(
                f"{path} was written for a different dataset ({detail}). Move it away "
                f"to start over, or point dataset.train_data_path back at that dataset."
            )

        expected_len = (self.size + 7) // 8
        published, counts = {}, {}
        for key, entry in state.get("open_epochs", {}).items():
            bitmap = bytearray(zlib.decompress(base64.b64decode(entry["bitmap_zlib_b64"])))
            if len(bitmap) != expected_len:
                raise ProgressMismatchError(
                    f"{path}: epoch {key} bitmap is {len(bitmap)} bytes, expected {expected_len}"
                )
            count = sum(bin(b).count("1") for b in bitmap)
            if count != entry.get("published"):
                raise ProgressMismatchError(
                    f"{path}: epoch {key} bitmap has {count} rows set, "
                    f"file says {entry.get('published')}"
                )
            published[int(key)] = bitmap
            counts[int(key)] = count

        self._published, self._counts = published, counts
        self.last_epoch = state.get("last_epoch")
        self.loop_state = dict(state.get("loop_state") or {})
        self.resumed = True
        owed = sum(self.size - c for c in counts.values())
        logger.info(
            "Resuming generation from %s (saved by job %s): last epoch %s, open epochs %s, "
            "%d rows still owed",
            path, state.get("job_id"), self.last_epoch, self.open_epochs, owed,
        )
