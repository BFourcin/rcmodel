"""
Tests for model-level recording (evaluate(record_window=...)) and the plots drawn from it.

The fixture building has 9 rooms, so everything here runs multi-room - the case the old
single-room plotting got wrong.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rcmodel import (
    PARAM_KEYS,
    best_records_over_time,
    env_creator,
    evaluate,
    load_model_record,
    make_dataloaders,
    plot_model_record,
    plot_residual_heatmap,
    save_model_record,
)


class AlternatingPolicy:
    """Stands in for an RLlib algorithm: cooling off, on, off, on, ... - so both actions appear."""

    def __init__(self):
        self.calls = 0

    def compute_single_action(self, obs, explore=None):
        self.calls += 1
        return self.calls % 2


@pytest.fixture
def env_and_eval_loader(get_model_config, data_config, env_config):
    env = env_creator({**env_config, "model_config": get_model_config})
    _, eval_dataloader = make_dataloaders(data_config)
    return env, eval_dataloader


@pytest.fixture
def record(env_and_eval_loader):
    env, eval_dataloader = env_and_eval_loader
    _, record = evaluate(env, AlternatingPolicy(), eval_dataloader, record_window=0)
    return record


def test_recording_is_off_by_default(env_and_eval_loader):
    env, eval_dataloader = env_and_eval_loader
    _, record = evaluate(env, AlternatingPolicy(), eval_dataloader)
    assert record is None


def test_record_matches_what_was_scored(env_and_eval_loader, get_model_config):
    """The record is the scored episode itself, not a re-run, and recording changes nothing."""
    env, eval_dataloader = env_and_eval_loader
    rewards, record = evaluate(env, AlternatingPolicy(), eval_dataloader, record_window=0)
    rewards_unrecorded, _ = evaluate(env, AlternatingPolicy(), eval_dataloader)

    np.testing.assert_allclose(rewards, rewards_unrecorded)
    assert record["window_reward"] == pytest.approx(rewards[0])
    assert record["step_reward"].sum() == pytest.approx(record["window_reward"])

    n_steps = len(record["action"])
    assert len(record["action_start"]) == len(record["action_end"]) == len(record["step_reward"]) == n_steps
    assert set(record["action"].tolist()) == {0, 1}
    np.testing.assert_array_equal(record["action_end"][:-1], record["action_start"][1:])

    time = record["time"]
    assert np.all(np.diff(time) > 0), "trajectory rows must be strictly in time order, with no duplicates"
    assert time[0] == record["measured_time"][0]
    assert time[-1] == record["measured_time"][-1]

    room_names = get_model_config["room_names"]
    assert record["room_names"].tolist() == room_names
    assert record["states"].shape == (len(time), 2 + len(room_names))
    assert record["measured"].shape == (len(record["measured_time"]), len(room_names))
    assert record["param_names"].tolist() == list(PARAM_KEYS)


def test_record_round_trips_through_npz(record, tmp_path):
    path = tmp_path / "record.npz"
    save_model_record(path, record)
    loaded = load_model_record(path)

    assert set(loaded) == set(record)
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(loaded[key], value)
        else:
            assert loaded[key] == value


def test_room_names_must_match_data_columns(get_model_config, env_config):
    """Rooms are matched to data columns by position, so a reordered room list must fail
    loudly rather than score each room against another room's measurements."""
    reordered = {**get_model_config, "room_names": list(reversed(get_model_config["room_names"]))}
    with pytest.raises(ValueError, match="do not match"):
        env_creator({**env_config, "model_config": reordered})


def test_plot_model_record_draws_every_room(record):
    fig = plot_model_record(record)
    try:
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        for name in record["room_names"]:
            assert sum(title.startswith(f"{name} ") for title in titles) == 1, f"no single panel for room {name}"
        assert len(fig.axes) == len(record["room_names"]) + 2  # plus the outdoor and net-heat strips
    finally:
        plt.close(fig)


def test_plot_residual_heatmap_has_a_row_per_room(record):
    fig = plot_residual_heatmap(record)
    try:
        heatmap = next(ax for ax in fig.axes if ax.images)
        assert [label.get_text() for label in heatmap.get_yticklabels()] == record["room_names"].tolist()
        assert heatmap.images[0].get_array().shape == (len(record["room_names"]), len(record["time"]))
    finally:
        plt.close(fig)


def test_best_records_over_time(tmp_path):
    """At each evaluation, the best of every trial's latest record - and only when it changes."""
    evaluations = [("a", 1.0, -10.0), ("b", 2.0, -5.0), ("a", 3.0, -20.0), ("a", 4.0, -1.0)]
    for trial_id, timestamp, score in evaluations:
        (tmp_path / trial_id).mkdir(exist_ok=True)
        save_model_record(
            tmp_path / trial_id / f"{timestamp}.npz", {"trial_id": trial_id, "timestamp": timestamp, "score": score}
        )

    sequence = best_records_over_time(tmp_path)
    # a@3 scores worse than b's latest, so b stays best and nothing new is appended.
    assert [(r["trial_id"], r["timestamp"]) for r in sequence] == [("a", 1.0), ("b", 2.0), ("a", 4.0)]

    thinned = best_records_over_time(tmp_path, n=2)
    assert [(r["trial_id"], r["timestamp"]) for r in thinned] == [("a", 1.0), ("a", 4.0)]
