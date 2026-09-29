"""The start of a file: get_iv_array's spin-up of the latent nodes (warmup_cycles) and the dataloaders leaving the
file's first window out of training and evaluation (skip_start_windows)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rcmodel import (
    BuildingTemperatureDataset,
    make_dataloaders,
    model_creator,
    physical_to_scaled,
    slowest_time_constant_days,
    training_windows_dataloader,
)
from rcmodel.rc_model import _room_mass_history, _spin_up, get_iv_array, steady_state_iv

DT = 600  # s
WINDOW = 432  # rows: 72 h at 10 min, as the real cases


# --------------------------------------------------------------------------- spin-up


def test_spin_up_iterates_and_stops_once_settled():
    """x <- cycle_end(x): the given number of cycles at most, and no more once a cycle changes x by < tol."""
    calls = []

    def halve_towards_two(x):
        calls.append(x.copy())
        return 0.5 * x + 1.0  # fixed point 2

    np.testing.assert_allclose(_spin_up(halve_towards_two, np.zeros(1), cycles=5), [1.9375])
    assert len(calls) == 5
    calls.clear()
    np.testing.assert_allclose(_spin_up(halve_towards_two, np.zeros(1), cycles=100, tol=0.01), [2.0], atol=0.01)
    assert len(calls) < 15
    np.testing.assert_array_equal(_spin_up(halve_towards_two, np.zeros(1), cycles=0), [0.0])


@pytest.fixture
def heavy_building(get_model_config, fake_rooms, tmp_path):
    """A slow building (time constant of days) over 12 days of daily weather and room temperature warming by a
    degree a day, as a season does: the steady-state guess from the whole history's means is far from the walls'
    state at the start of the file, and a slow building takes days to forget it."""
    n_rows = 12 * 144
    t = 1_600_000_000 + np.arange(n_rows) * DT
    day = 2 * np.pi * (t - t[0]) / 86400
    trend = (t - t[0]) / 86400  # degC per day
    config = dict(get_model_config)
    config.update(
        {
            "weather_data_outdoor_temperature": 4 + trend + 6 * np.sin(day),
            "weather_data_UTC_time": t,
            "weather_data_ghi": np.clip(700 * np.sin(day), 0, None),
            "k_sa": [0.5, 0.5],
        }
    )
    physical = {
        "C_rm": 5e3,
        "C1": 5e6,
        "C2": 5e6,
        "R1": 1.0,
        "R2": 2.0,
        "R3": 1.0,
        "Rin": 0.5,
        "k_sa": 0.5,
        "cool": 0.0,
        "gain": 0.0,
        "solar": 0.0,
    }
    config["parameters"] = physical_to_scaled(config, physical)
    tau = slowest_time_constant_days(config, physical)
    assert 1 < tau < 10, f"fixture should be a slow building, tau {tau:.2f} d"

    names, _ = fake_rooms
    room = 20 + 0.3 * trend + 1.5 * np.sin(day - 1.0)
    df = pd.DataFrame(np.repeat(room[:, None], len(names), axis=1), columns=names)
    df.insert(0, "time", t)
    df.insert(0, "date-time", pd.to_datetime(t, unit="s"))
    csv_path = tmp_path / "heavy_sorted.csv"
    df.to_csv(csv_path, index=False)
    dataset = BuildingTemperatureDataset(csv_path, WINDOW, all=True, block_indices=[1, 2, 3])
    return config, dataset, t


def _iv_at(config, dataset, cycles, when):
    model = model_creator({**config, "warmup_cycles": cycles})
    model._build_matrices()
    model._build_loads()
    return get_iv_array(model, dataset)(torch.tensor(float(when)))[0:2], model


def test_spin_up_starts_the_file_near_its_periodic_state(heavy_building):
    """With a time constant of days, walking the history from the steady-state guess leaves the walls off at the
    start of the second window; five spin-up cycles over the first window bring them close to the settled state
    (a long spin-up)."""
    config, dataset, t = heavy_building
    start_of_window_1 = t[WINDOW]
    settled, _ = _iv_at(config, dataset, 60, start_of_window_1)
    no_spin, _ = _iv_at(config, dataset, 0, start_of_window_1)
    spun, _ = _iv_at(config, dataset, 5, start_of_window_1)
    error_without = (no_spin - settled).abs().max().item()
    error_with = (spun - settled).abs().max().item()
    assert error_without > 0.2, f"fixture too fast to show the warm-up error ({error_without:.3f} degC)"
    assert error_with < 0.25 * error_without, (error_with, error_without)


def test_no_spin_up_starts_from_the_whole_history_guess(heavy_building):
    """warmup_cycles = 0 is the old behaviour: the history walk starts at the steady-state guess for the mean
    sol-air and room temperature over the whole history."""
    config, dataset, t = heavy_building
    first, model = _iv_at(config, dataset, 0, t[0])
    t_hist, temps = dataset.get_history()
    tout = torch.as_tensor(model.sol_air_temperature(t_hist.squeeze())).mean()
    tin = temps[:, 0].mean()
    guess = steady_state_iv(model, tout, tin).squeeze()[0:2]
    torch.testing.assert_close(first, guess, atol=1e-4, rtol=0)


def test_room_mass_spin_up():
    """The mass node low-pass: spun up over the first rows it starts at their periodic state, not the first air
    temperature, and with no cycles it starts at the first air temperature as before."""
    building = SimpleNamespace(rooms=[0], r_rm_mass=0.5, c_rm_mass=4e5)  # tau 2.3 d
    t = np.arange(10 * 144) * DT
    air = (22 + 2 * np.sin(2 * np.pi * t / 86400))[:, None]
    np.testing.assert_allclose(_room_mass_history(building, t, air)[0], air[0])
    long = _room_mass_history(building, t, air, cycles=200, n_spin=144)[0]
    spun = _room_mass_history(building, t, air, cycles=5, n_spin=144)[0]
    assert abs(spun - long) < 0.25 * abs(air[0] - long)


# --------------------------------------------------------------------------- skip_start_windows


def _block(batch, t0, size, dt):
    return (batch[0][0, 0].item() - t0) / (size * dt)


def test_first_window_is_never_trained_or_scored(data_config):
    """Default skip_start_windows = 1, every split mode: random training windows never start inside block 0, and
    neither the deterministic training windows nor the evaluation windows include it."""
    size = data_config["sample_size"] // 2  # 12 blocks over the fixture's 6 hours
    dt = data_config["dt"]
    modes = {
        "tail": "tail",
        "interleaved": {"mode": "interleaved", "every": 3},
        "blocks": {"mode": "blocks", "train": [0, 1, 2, 4], "eval": [3, 5]},
    }
    for name, split in modes.items():
        config = {**data_config, "sample_size": size, "eval_split": split}
        train_windows = training_windows_dataloader(config)
        train_random, evaluation = make_dataloaders(config)
        t0 = train_windows.dataset.get_history()[0][0].item()
        assert min(round(_block(b, t0, size, dt)) for b in train_windows) >= 1, name
        assert min(_block(b, t0, size, dt) for b in evaluation) >= 1, name
        batches = iter(train_random)
        assert min(_block(next(batches), t0, size, dt) for _ in range(40)) >= 1 - 1e-9, name


def test_skip_zero_keeps_the_old_windows(data_config):
    """skip_start_windows = 0 reproduces the old tail training windows exactly."""
    config = {**data_config, "skip_start_windows": 0}
    loader = training_windows_dataloader(config)
    old = BuildingTemperatureDataset(config["csv_path"], config["sample_size"], all=False, train=True, test=False)
    assert [b[0][0, 0].item() for b in loader] == [old[i][0][0].item() for i in range(len(old))]
    skipped = training_windows_dataloader(data_config)
    assert [b[0][0, 0].item() for b in skipped] == [old[i][0][0].item() for i in range(1, len(old))]


def test_skipping_everything_is_an_error(data_config):
    with pytest.raises(ValueError, match="skipping"):
        training_windows_dataloader({**data_config, "skip_start_windows": 50})
    with pytest.raises(ValueError, match="skip_start_windows"):
        make_dataloaders({**data_config, "skip_start_windows": -1})
