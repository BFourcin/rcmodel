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
import torch

from rcmodel import (
    H_OUT,
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


def test_record_carries_the_heat_input_that_drove_it(env_and_eval_loader):
    """ghi, solar_w and net_heat_w line up with the simulated rows and follow from the
    parameters: net heat = floor area * (gain + solar * GHI - cool * action), all rooms, with
    the action in force at each row."""
    env, eval_dataloader = env_and_eval_loader
    _, record = evaluate(env, AlternatingPolicy(), eval_dataloader, record_window=0)
    n_rows = len(record["time"])
    area = record["room_area"]

    assert record["ghi"].shape == record["solar_w"].shape == record["net_heat_w"].shape == (n_rows,)
    assert record["solar_p"].shape == area.shape
    assert record["ghi"].max() > 0, "the fixture GHI should be non-zero in the evaluation window"

    model = env.unwrapped.RC
    np.testing.assert_allclose(record["ghi"], model.ghi(torch.tensor(record["time"])))
    np.testing.assert_allclose(record["solar_w"], record["ghi"] * np.sum(record["solar_p"] * area))

    # The action in force at each row, rebuilt from the step boundaries.
    step = np.searchsorted(record["action_end"], record["time"], side="left")
    action = record["action"][np.minimum(step, len(record["action"]) - 1)]
    expected = np.sum(record["gain_w"]) + record["solar_w"] - action * np.sum(record["cool_w"])
    np.testing.assert_allclose(record["net_heat_w"], expected, rtol=1e-6)  # loads are float32


def test_record_carries_the_sol_air_temperature(env_and_eval_loader):
    """sol_air_temperature is outdoor + k_sa * GHI / H_OUT at every simulated row."""
    env, eval_dataloader = env_and_eval_loader
    _, record = evaluate(env, AlternatingPolicy(), eval_dataloader, record_window=0)
    k_sa = record["param_values"][list(record["param_names"]).index("k_sa")]
    assert k_sa > 0, "the fixture should have a sol-air effect to check"

    assert record["sol_air_temperature"].shape == record["outdoor"].shape
    np.testing.assert_allclose(record["sol_air_temperature"], record["outdoor"] + k_sa * record["ghi"] / H_OUT, rtol=1e-6)


def test_record_from_before_sol_air_still_plots(record):
    """Records written by older runs have no sol_air_temperature and no k_sa."""
    old = {key: value for key, value in record.items() if key != "sol_air_temperature"}
    names = list(record["param_names"])
    keep = [i for i, name in enumerate(names) if name != "k_sa"]
    old["param_names"] = record["param_names"][keep]
    old["param_values"] = record["param_values"][keep]
    fig = plot_model_record(old)
    try:
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert "sol-air" not in labels
    finally:
        plt.close(fig)


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


def test_plot_model_record_draws_the_solar_gain(record):
    fig = plot_model_record(record)
    try:
        ax_heat = fig.axes[-1]
        labels = [line.get_label() for line in ax_heat.get_lines()]
        assert "solar" in labels and "net" in labels
        solar_line = next(line for line in ax_heat.get_lines() if line.get_label() == "solar")
        np.testing.assert_allclose(solar_line.get_ydata(), record["solar_w"])
        assert "solar p" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_record_from_before_solar_still_plots(record):
    """Records written by older runs have no ghi/solar_w/net_heat_w/solar_p."""
    old = {key: value for key, value in record.items() if key not in ("ghi", "solar_w", "net_heat_w", "solar_p")}
    fig = plot_model_record(old)
    try:
        assert len(fig.axes) == len(record["room_names"]) + 2
        assert "solar p" not in fig._suptitle.get_text()
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


def test_thermostat_record_carries_the_cooling_it_delivered(get_model_config, data_config, env_config):
    """In thermostat mode the cooling is worked out row by row, so the record must carry what was
    actually delivered - never more than capacity, never heating, only while the action is on -
    and net heat must include exactly that."""
    config = {**get_model_config, "cooling_mode": "thermostat", "T_set": [21, 21], "gain": [0, 200]}
    env = env_creator({**env_config, "model_config": config})
    _, eval_dataloader = make_dataloaders(data_config)
    _, record = evaluate(env, AlternatingPolicy(), eval_dataloader, record_window=0)

    delivered = record["delivered_cooling_w"]
    assert record["cooling_mode"] == "thermostat"
    assert delivered.shape == (len(record["time"]), len(record["room_names"]))
    assert (delivered <= 0).all() and (delivered >= -record["cool_w"][None, :] * (1 + 1e-6)).all()

    step = np.searchsorted(record["action_end"], record["time"], side="left")
    action = record["action"][np.minimum(step, len(record["action"]) - 1)]
    assert np.all(delivered[action == 0] == 0)
    assert (delivered[action == 1] < 0).any(), "the fixture should need some cooling"
    np.testing.assert_allclose(
        record["net_heat_w"], np.sum(record["gain_w"]) + record["solar_w"] + delivered.sum(axis=1), rtol=1e-6
    )
    assert list(record["state_names"][: record["n_latent"]]) == ["Te1", "Te2"]
    assert list(record["state_names"][record["n_latent"] :]) == list(record["room_names"])


def test_room_mass_env_scores_and_records_only_the_rooms(get_model_config, data_config, env_config):
    """With mass nodes after the rooms, the reward and the policy's observation must still be the
    rooms, and the record must name every state."""
    config = {**get_model_config, "room_mass": True, "C_rm": [1, 1], "C_rm_mass": [1e3, 1e5], "R_rm_mass": [0.1, 5]}
    env = env_creator({**env_config, "model_config": config})
    n_rooms = len(config["room_names"])
    obs, _ = env.reset()
    assert len(obs) == n_rooms + 4  # rooms + sin/cos of day and week
    _, eval_dataloader = make_dataloaders(data_config)
    _, record = evaluate(env, AlternatingPolicy(), eval_dataloader, record_window=0)

    assert record["states"].shape[1] == 2 + 2 * n_rooms
    names = list(record["state_names"])
    assert names[2 : 2 + n_rooms] == list(record["room_names"])
    assert names[2 + n_rooms :] == [f"{room} mass" for room in record["room_names"]]
    # The residuals the plots draw are room-by-room, not mass nodes.
    from rcmodel.tools.plotting import room_rmse

    assert room_rmse(record).shape == (n_rooms,)
    plot_model_record(record)
    plt.close("all")
