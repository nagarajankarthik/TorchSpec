"""Tests for resuming hidden-state generation after a killed job."""

import json
import os
import signal
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchspec.controller.generation_progress import (
    GenerationProgress,
    ProgressMismatchError,
    check_progress_file,
    resolve_progress_path,
)

IDS = [f"sample_{i}" for i in range(20)]


# --- GenerationProgress -------------------------------------------------------


def test_fresh_run_starts_at_epoch_zero():
    assert GenerationProgress(IDS).resume_epochs() == [0]


def test_partial_epoch_survives_a_round_trip(tmp_path):
    path = str(tmp_path / "p.json")
    progress = GenerationProgress(IDS, fingerprint={"train_data_path": "/d"})
    progress.open_epoch(0)
    for data_id in IDS[::3]:
        progress.mark_published(data_id)
    progress.loop_state = {"completed_steps": 7}
    progress.save(path)

    restored = GenerationProgress(IDS, fingerprint={"train_data_path": "/d"})
    restored.load(path)
    assert restored.resumed
    assert restored.resume_epochs() == [0]
    assert restored.loop_state == {"completed_steps": 7}
    assert [restored.is_published(0, d) for d in IDS] == [d in IDS[::3] for d in IDS]
    assert restored.published_in(0) == len(IDS[::3])


def test_finished_epoch_closes_and_the_next_one_starts():
    progress = GenerationProgress(IDS)
    progress.open_epoch(0)
    for data_id in IDS:
        progress.mark_published(data_id)
    assert progress.open_epochs == []
    assert progress.published_in(0) == len(IDS)
    assert progress.resume_epochs() == [1]


def test_overlapping_epochs_credit_the_oldest_first():
    progress = GenerationProgress(IDS)
    progress.open_epoch(0)
    progress.open_epoch(1)
    progress.mark_published("sample_4")
    progress.mark_published("sample_4")
    assert progress.is_published(0, "sample_4")
    assert progress.is_published(1, "sample_4")
    progress.mark_published("sample_4")
    assert progress.unattributed == 1


def test_unknown_ids_are_counted_not_tracked():
    progress = GenerationProgress(IDS)
    progress.open_epoch(0)
    progress.mark_published("not_in_dataset")
    progress.mark_published(None)
    assert progress.unknown_ids == 2
    assert progress.published_in(0) == 0


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        GenerationProgress(["a", "b", "a"])


@pytest.mark.parametrize(
    "ids, fingerprint",
    [
        (IDS[:-1], {"train_data_path": "/d"}),  # a row filtered out
        (IDS[::-1], {"train_data_path": "/d"}),  # rows renumbered
        (IDS, {"train_data_path": "/other"}),  # different dataset
    ],
)
def test_progress_for_another_dataset_is_refused(tmp_path, ids, fingerprint):
    path = str(tmp_path / "p.json")
    GenerationProgress(IDS, fingerprint={"train_data_path": "/d"}).save(path)
    with pytest.raises(ProgressMismatchError):
        GenerationProgress(ids, fingerprint=fingerprint).load(path)


def test_corrupt_bitmap_is_refused(tmp_path):
    path = tmp_path / "p.json"
    progress = GenerationProgress(IDS)
    progress.open_epoch(0)
    progress.mark_published("sample_1")
    progress.save(str(path))
    state = json.loads(path.read_text())
    state["open_epochs"]["0"]["published"] = 5
    path.write_text(json.dumps(state))
    with pytest.raises(ProgressMismatchError, match="rows set"):
        GenerationProgress(IDS).load(str(path))


def test_failed_save_leaves_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    progress = GenerationProgress(IDS)
    progress.open_epoch(0)
    progress.save(str(path))
    before = path.read_text()

    progress.mark_published("sample_2")
    monkeypatch.setattr(json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError):
        progress.save(str(path))
    assert path.read_text() == before
    assert os.listdir(tmp_path) == ["p.json"]


def test_existing_progress_needs_resume_to_be_explicit(tmp_path):
    path = tmp_path / "p.json"
    check_progress_file(str(path), resume=False)  # nothing there yet
    path.write_text("{}")
    with pytest.raises(FileExistsError, match="resume_generation"):
        check_progress_file(str(path), resume=False)
    check_progress_file(str(path), resume=True)
    check_progress_file(None, resume=False)  # untracked: warns only


def test_progress_path_defaults_to_output_dir(tmp_path):
    out = str(tmp_path / "out")
    assert resolve_progress_path(SimpleNamespace(output_dir=out)) == os.path.join(
        out, "generation_progress.json"
    )
    explicit = str(tmp_path / "elsewhere.json")
    args = SimpleNamespace(output_dir=out, generation_progress_file=explicit)
    assert resolve_progress_path(args) == explicit
    assert resolve_progress_path(SimpleNamespace(output_dir="")) is None


# --- controller -----------------------------------------------------------------


@dataclass
class ControllerArgs:
    seed: int = 0
    shuffle_dataset: bool = True
    length_group_size: int = 4
    train_data_path: str = "/d"
    redis_dispatch_batch_size: int = 1


@pytest.fixture
def make_controller(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "6379")
    from torchspec.controller.training_controller import AsyncTrainingController

    def make(path, *, resume):
        actor = AsyncTrainingController(ControllerArgs())
        actor._stored_dataset = [{"data_id": d, "seq_len": (i * 7) % 13 + 1} for i, d in enumerate(IDS)]
        actor.enable_progress(path, resume=resume)
        return actor

    return make


def _publish(actor, data_ids, *, fail=(), stream=None):
    """Run data_ids through _dispatch_to_redis with XADD failing for ``fail``."""
    from torchspec.controller.training_controller import MooncakeEntry
    from torchspec.utils.types import InferenceOutput

    outputs = []
    for data_id in data_ids:
        key = f"key-{data_id}"
        actor._mooncake_entries[key] = MooncakeEntry(
            inserted_at=time.time(), dispatched_at=time.time(), num_bytes=1,
            has_last_hidden_states=True, has_target=False,
        )
        outputs.append(InferenceOutput(data_id=data_id, mooncake_key=key, tensor_shapes={}))
    actor._xadd_batch = lambda samples, stream: [s for s in samples if s.data_id in fail]
    actor._publish_retry_seconds = 0
    return actor._dispatch_to_redis(outputs, stream or actor.redis_train_stream)


def _buffer_ids(actor):
    return [entry.data_id for entry in actor.prompt_buffer]


def test_restarted_job_owes_exactly_the_unpublished_rows(tmp_path, make_controller):
    path = str(tmp_path / "p.json")
    first = make_controller(path, resume=False)
    first.submit_training_dataset()
    epoch_order = _buffer_ids(first)

    # Completion order differs from submission order, and one XADD fails.
    done = epoch_order[5:12:2] + epoch_order[:3]
    assert _publish(first, done, fail={epoch_order[1]}) == len(done) - 1
    first.save_progress({"completed_steps": len(done) - 1})

    second = make_controller(path, resume=True)
    second.submit_training_dataset()
    published = set(done) - {epoch_order[1]}
    assert _buffer_ids(second) == [d for d in epoch_order if d not in published]
    assert second.get_resume_state() == {
        "completed_steps": len(done) - 1,
        "epoch": 0,
        "published_in_epoch": len(published),
    }


def test_resume_follows_the_newest_epoch_started(tmp_path, make_controller):
    path = str(tmp_path / "p.json")
    first = make_controller(path, resume=False)
    first.submit_training_dataset()
    first.reload_dataset()  # epoch 1 overlaps the tail of epoch 0
    _publish(first, IDS)  # credited to epoch 0, which closes
    _publish(first, IDS[:5])  # epoch 1 partly done
    first.save_progress({"completed_steps": 25})

    second = make_controller(path, resume=True)
    second.submit_training_dataset()
    assert sorted(_buffer_ids(second)) == sorted(IDS[5:])
    assert second._dataset_epoch == 1
    second.reload_dataset()
    assert second._dataset_epoch == 2


def test_eval_stream_does_not_count_as_progress(tmp_path, make_controller):
    actor = make_controller(str(tmp_path / "p.json"), resume=False)
    actor.submit_training_dataset()
    assert _publish(actor, IDS[:3], stream=actor.redis_eval_stream) == 3
    assert actor._progress.published_in(0) == 0


# --- loop -----------------------------------------------------------------------


class FakeController:
    dispatch_batch_size = 1

    def __init__(self, *, completed_steps=0, on_dispatch=None):
        self.resume = {"completed_steps": completed_steps, "epoch": 0, "published_in_epoch": 0}
        self.on_dispatch = on_dispatch
        self.dispatches = 0
        self.saves = []

    def submit_training_dataset(self):
        return 10

    def get_resume_state(self):
        return dict(self.resume)

    def get_prompt_buffer_size(self):
        return 1

    def start_eviction_sweeper(self):
        pass

    def reload_dataset(self):
        pass

    def try_dispatch_batch(self):
        self.dispatches += 1
        if self.on_dispatch:
            self.on_dispatch(self.dispatches)
        return True

    def save_progress(self, loop_state):
        self.saves.append(dict(loop_state))

    def get_full_status(self):
        return {"elapsed_seconds": 0.0, "avg_inference_speed": 0.0, "avg_training_speed": 0.0}


class BusyInference:
    def get_status(self):
        return {"prompt_buffer_size": 1, "pending_tasks": 0, "current_pool_size": 0}


def _loop_args(**kw):
    return SimpleNamespace(num_epochs=1, generation_progress_save_seconds=0, **kw)


def test_loop_continues_from_the_saved_step():
    from torchspec.controller.loop import training_loop

    controller = FakeController(completed_steps=6)
    training_loop(_loop_args(), controller, BusyInference(), None, dataset_size=10)
    assert controller.dispatches == 4
    assert controller.saves[-1] == {"completed_steps": 10}


def test_sigterm_saves_progress_before_exiting():
    from torchspec.controller.loop import training_loop

    def kill_on_third(n):
        if n == 3:
            os.kill(os.getpid(), signal.SIGTERM)

    before = signal.getsignal(signal.SIGTERM)
    controller = FakeController(on_dispatch=kill_on_third)
    with pytest.raises(SystemExit) as exc:
        training_loop(_loop_args(), controller, BusyInference(), None, dataset_size=10)
    assert exc.value.code == 128 + signal.SIGTERM
    # The third dispatch went out before the handler ran; it was not counted.
    assert controller.saves == [{"completed_steps": 2}]
    assert signal.getsignal(signal.SIGTERM) is before
