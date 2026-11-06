"""Tests for length grouping."""

from dataclasses import dataclass

import pytest
import torch

from torchspec.data.utils import length_grouped_order, sample_length_hint


def samples(*lengths):
    return [{"data_id": f"s{i}", "seq_len": n} for i, n in enumerate(lengths)]


def lengths_of(entries):
    return [sample_length_hint(e) for e in entries]


@pytest.mark.parametrize(
    "sample, expected",
    [
        ({"seq_len": 512}, 512),  # offline replay row
        ({"input_ids": torch.zeros(128), "formatted_prompt": "ignored"}, 128),  # tokenized
        ({"formatted_prompt": "hello"}, 5),  # defer_tokenization: chars stand in
    ],
)
def test_length_hint_reads_whichever_field_is_present(sample, expected):
    assert sample_length_hint(sample) == expected


def test_sorts_longest_first_within_each_chunk():
    # Chunks are positional, so the 100 in the second chunk stays there.
    assert lengths_of(length_grouped_order(samples(1, 2, 100, 3), 2)) == [2, 1, 100, 3]


def test_trailing_partial_chunk_is_sorted_too():
    assert lengths_of(length_grouped_order(samples(1, 2, 3, 4, 9, 5), 4)) == [4, 3, 2, 1, 9, 5]


def test_small_dataset_becomes_a_full_sort():
    assert lengths_of(length_grouped_order(samples(1, 5, 3), 1024)) == [5, 3, 1]


@pytest.mark.parametrize("group_size", [0, 1])
def test_group_size_of_one_or_less_disables_grouping(group_size):
    entries = samples(1, 5, 3)
    assert length_grouped_order(entries, group_size) == entries


def test_leaves_the_input_list_untouched():
    entries = samples(1, 5, 3)
    length_grouped_order(entries, 3)
    assert lengths_of(entries) == [1, 5, 3]


def test_groups_similar_lengths_into_neighbouring_dispatches():
    # The property that matters: a dispatch-sized window should span a narrow
    # range of lengths after grouping.
    entries = samples(*((i * 37) % 1000 for i in range(600)))

    def mean_spread(ordered, dispatch=4):
        windows = [lengths_of(ordered[i : i + dispatch]) for i in range(0, 600, dispatch)]
        return sum(max(w) - min(w) for w in windows) / len(windows)

    assert mean_spread(length_grouped_order(entries, 128)) < mean_spread(entries) / 5


# --- controller wiring ------------------------------------------------------


@dataclass
class MockArgs:
    per_dp_rank_batch_size: int = 1
    max_sample_pool_size: int = 0
    seed: int = 0
    shuffle_dataset: bool = True
    length_group_size: int = 8


@pytest.fixture(autouse=True)
def _redis_env(monkeypatch):
    # The controller builds a Redis client at construction; it connects lazily,
    # so nothing needs to be listening.
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "6379")


def controller(dataset, **overrides):
    from torchspec.controller.training_controller import AsyncTrainingController

    actor = AsyncTrainingController(MockArgs(**overrides))
    actor._stored_dataset = dataset
    return actor


SHUFFLE_ME = samples(*((i * 13) % 100 for i in range(32)))


def test_grouping_runs_after_the_shuffle():
    data = controller(SHUFFLE_ME)._prepare_dataset()

    assert len(data) == 32
    for start in range(0, 32, 8):
        chunk = lengths_of(data[start : start + 8])
        assert chunk == sorted(chunk, reverse=True)


def test_disabled_grouping_leaves_the_shuffled_order():
    grouped = controller(SHUFFLE_ME)._prepare_dataset()
    plain = controller(SHUFFLE_ME, length_group_size=0)._prepare_dataset()
    assert lengths_of(grouped) != lengths_of(plain)


def test_same_epoch_rebuilds_the_same_order():
    # Mid-epoch resume slices this order, so it has to be reproducible.
    first = controller(SHUFFLE_ME)._prepare_dataset()
    second = controller(SHUFFLE_ME)._prepare_dataset()
    assert [e["data_id"] for e in first] == [e["data_id"] for e in second]


def test_exclude_published_keeps_the_grouped_order(tmp_path):
    actor = controller(SHUFFLE_ME)
    actor.enable_progress(str(tmp_path / "progress.json"), resume=False)
    actor._progress.open_epoch(0)
    full = actor._prepare_dataset()
    published = {e["data_id"] for e in full[::3]}
    for data_id in published:
        actor._progress.mark_published(data_id)

    remaining = actor._prepare_dataset(exclude_published=True)
    assert remaining == [e for e in full if e["data_id"] not in published]


def test_each_epoch_groups_different_samples_together():
    actor = controller(SHUFFLE_ME)
    epoch0 = actor._prepare_dataset()
    actor._dataset_epoch = 1
    epoch1 = actor._prepare_dataset()

    assert {e["data_id"] for e in epoch0[:8]} != {e["data_id"] for e in epoch1[:8]}
