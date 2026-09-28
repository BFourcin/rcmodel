"""
Tests for the schedule controller (SchedulePolicy), the optional observation features of PreprocessEnv, and
training_windows_dataloader.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from rcmodel import (
    DEFAULT_OBSERVATION_FEATURES,
    SchedulePolicy,
    env_creator,
    evaluate,
    free_float_mask,
    make_dataloaders,
    preprocess_observation,
    schedule_available,
    training_windows_dataloader,
)
from rcmodel.optimisation import is_weekday, observation_features

WEEKDAYS = [0, 1, 2, 3, 4]


def _unix(stamp):
    return pd.Timestamp(stamp).timestamp()


# --------------------------------------------------------------------------- schedule


def test_weekday_is_monday_aligned():
    # 1995-06-05 was a Monday, 1995-06-10 a Saturday.
    assert is_weekday(_unix("1995-06-05 12:00"))
    assert is_weekday(_unix("1995-06-09 23:59"))
    assert not is_weekday(_unix("1995-06-10 00:00"))
    assert not is_weekday(_unix("1995-06-11 12:00"))
    # A UTC offset moves the day boundary: Friday 23:30 UTC is Saturday 00:30 at UTC+1.
    assert not is_weekday(_unix("1995-06-09 23:30"), utc_offset_hours=1)


def test_schedule_available_matches_free_float_mask():
    """One window is exactly the complement of the hvac_schedule free-float mask."""
    time = np.arange(_unix("1995-06-01"), _unix("1995-06-15"), 600.0)
    schedule = {"weekdays": WEEKDAYS, "start": "07:00", "end": "19:00"}
    assert np.array_equal(schedule_available(time, schedule), ~free_float_mask(time, schedule))
    numeric = {"weekdays": WEEKDAYS, "start": 7, "end": 19}
    assert np.array_equal(schedule_available(time, numeric), ~free_float_mask(time, schedule))


def test_schedule_windows_combine_and_collapse():
    time = np.array([_unix("1995-06-10 10:00"), _unix("1995-06-05 06:30"), _unix("1995-06-05 07:00")])
    weekend = {"weekdays": [5, 6], "start": 9, "end": 12}
    weekday = {"weekdays": WEEKDAYS, "start": 6.5, "end": 7}
    assert schedule_available(time, [weekend, weekday]).tolist() == [True, True, False]
    empty = {"weekdays": WEEKDAYS, "start": 12, "end": 12}
    assert not schedule_available(time, [empty]).any()
    assert not schedule_available(time, []).any()


class _FakeEnv:
    def __init__(self, time, room):
        import torch

        self.unwrapped = self
        self.observation = torch.tensor([[time, 20.0, 20.0, room]], dtype=torch.float64)


def test_schedule_policy_gate_and_setpoint():
    monday_noon = _unix("1995-06-05 12:00")
    saturday_noon = _unix("1995-06-10 12:00")
    window = {"weekdays": WEEKDAYS, "start": 7, "end": 19}
    assert SchedulePolicy(_FakeEnv(monday_noon, 25.0), window).compute_single_action(None) == 1
    assert SchedulePolicy(_FakeEnv(saturday_noon, 25.0), window).compute_single_action(None) == 0
    assert SchedulePolicy(_FakeEnv(monday_noon, 21.0), window, setpoint=22).compute_single_action(None) == 0
    assert SchedulePolicy(_FakeEnv(monday_noon, 23.0), window, setpoint=22).compute_single_action(None) == 1


def test_schedule_policy_evaluates(get_model_config, data_config, env_config):
    """evaluate() accepts it like an RLlib algorithm, and the score is repeatable."""
    _, eval_dataloader = make_dataloaders(data_config)
    env = env_creator({**env_config, "model_config": copy.deepcopy(get_model_config), "dataloader": eval_dataloader})
    policy = SchedulePolicy(env, {"weekdays": list(range(7)), "start": 0, "end": 24})
    first, _ = evaluate(env, policy, eval_dataloader)
    second, _ = evaluate(env, policy, eval_dataloader)
    assert first == second


# --------------------------------------------------------------------------- observation


def test_default_observation_unchanged(get_model_config, env_config):
    env = env_creator({**env_config, "model_config": copy.deepcopy(get_model_config)})
    n_rooms = env.unwrapped.n_rooms
    assert env.observation_space.shape == (n_rooms + 4,)
    obs, _ = env.reset()
    assert obs.shape == (n_rooms + 4,)
    x = np.array([21.0, 23.0][:n_rooms] + [22.0] * max(0, n_rooms - 2))
    t = _unix("1995-06-05 12:00")
    assert np.allclose(
        observation_features(x, t, 23.0, 1.5, DEFAULT_OBSERVATION_FEATURES), preprocess_observation(x, t, 23.0, 1.5)
    )


def test_extended_observation(get_model_config, env_config):
    features = ["temperature", "time_of_day", "weekday", "t_set", "t_minus_setpoint"]
    env = env_creator({**env_config, "model_config": copy.deepcopy(get_model_config), "observation_features": features})
    n_rooms = env.unwrapped.n_rooms
    size = n_rooms + 2 + 1 + 1 + n_rooms
    assert env.observation_space.shape == (size,)
    obs, _ = env.reset()
    assert obs.shape == (size,)
    assert env.observation_space.contains(obs)

    t_set = env.unwrapped.RC._t_set()
    mu, std = env.mu, env.std_dev
    x_norm = obs[:n_rooms]
    assert obs[n_rooms + 2] in (0.0, 1.0)
    assert obs[n_rooms + 3] == pytest.approx((t_set - mu) / std)
    # T_room - T_set, normalised, is consistent with the normalised temperature and setpoint.
    assert np.allclose(obs[n_rooms + 4 :], x_norm - (t_set - mu) / std)


def test_unknown_feature_rejected(get_model_config, env_config):
    with pytest.raises(ValueError, match="observation features"):
        env_creator({**env_config, "model_config": copy.deepcopy(get_model_config), "observation_features": ["nope"]})


# --------------------------------------------------------------------------- training windows


def test_training_windows_are_the_training_split(data_config):
    loader = training_windows_dataloader(data_config)
    _, eval_dataloader = make_dataloaders(data_config)
    train_times = {float(t) for batch in loader for t in batch[0].flatten()}
    eval_times = {float(t) for batch in eval_dataloader for t in batch[0].flatten()}
    assert train_times and not train_times & eval_times
    # Deterministic: walking it twice gives the same windows.
    assert [b[0][0, 0].item() for b in loader] == [b[0][0, 0].item() for b in loader]


def test_training_windows_interleaved(data_config):
    config = {**data_config, "sample_size": data_config["sample_size"] // 4, "eval_split": {"mode": "interleaved", "every": 3}}
    loader = training_windows_dataloader(config)
    _, eval_dataloader = make_dataloaders(config)
    train_starts = {b[0][0, 0].item() for b in loader}
    eval_starts = {b[0][0, 0].item() for b in eval_dataloader}
    assert train_starts and eval_starts and not train_starts & eval_starts
