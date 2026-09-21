"""
Baseline timing test for `rcmodel.rc_model.get_iv_array()`.

Production runs this on ~4 months of data at dt=30s (~350k rows), which with
the current implementation extrapolates to several minutes per call - too
slow to run as a test. Instead this parametrizes over small/fast dataset
sizes to get a repeatable timing baseline we can compare against once
get_iv_array() gets optimised.

Profiling (cProfile, n_rows=5760) shows ~95% of the time is spent inside
latent_f_ode's two Interp1D(...) calls (Tout_continuous, Tin_continuous),
which run once per RK4 sub-step (4 calls per step). Each call does a full
xitorch searchsorted/clamp/gather lookup for a *single* scalar time point,
rather than the whole trajectory being looked up in one batched call. That
per-call overhead, multiplied by every RK4 sub-step of every row, looks like
the main thing to fix rather than the ODE integration itself.
"""

import time

import numpy as np
import pandas as pd
import pytest
import torch

from rcmodel import BuildingTemperatureDataset, model_creator
from rcmodel.rc_model import get_iv_array

DT = 30  # seconds, matches production sampling rate


def _model_config(n_rows, fake_rooms):
    np.random.seed(42)
    fake_room_names, fake_room_coordinates = fake_rooms
    n_seconds = n_rows * DT
    t = np.arange(0, n_seconds, DT)
    outdoor = 10 + 5 * np.sin(2 * np.pi * t / n_seconds)

    return {
        "C_rm": [1e3, 1e5],
        "C1": [1e5, 1e8],
        "C2": [1e5, 1e8],
        "R1": [0.1, 5],
        "R2": [0.1, 5],
        "R3": [0.5, 6],
        "Rin": [0.1, 5],
        "cool": [0, 50],
        "gain": [0, 5],
        "room_names": fake_room_names,
        "room_coordinates": fake_room_coordinates,
        "weather_data_outdoor_temperature": outdoor,
        "weather_data_UTC_time": t,
        "cooling_policy": None,
        "load_model_path_policy": None,
        "load_model_path_physical": None,
        "parameters": {k: np.random.rand(1).item() for k in ["C_rm", "C1", "C2", "R1", "R2", "R3", "Rin", "cool", "gain"]},
    }


def _indoor_temperature_csv(tmp_path, n_rows, fake_rooms):
    """Synthetic per-room indoor-temperature CSV in the (date-time, time, <room columns>)
    format BuildingTemperatureDataset expects."""
    fake_room_names, _ = fake_rooms
    n_seconds = n_rows * DT
    t = np.arange(0, n_seconds, DT)
    rng = np.random.default_rng(7)
    base_indoor = 22 + 2 * np.sin(2 * np.pi * (t - 3 * 60**2) / n_seconds)
    room_offsets = rng.uniform(-1.0, 1.0, size=len(fake_room_names))
    indoor_temps = base_indoor[:, None] + room_offsets[None, :]

    df = pd.DataFrame(indoor_temps, columns=fake_room_names)
    df.insert(0, "time", t)
    df.insert(0, "date-time", pd.to_datetime(t, unit="s", origin="2021-01-01"))

    csv_path = tmp_path / f"indoor_temperature_{n_rows}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize("n_rows", [360, 1440, 5760])  # 3hrs, 12hrs, 2 days @ dt=30s
def test_get_iv_array_baseline_timing(n_rows, tmp_path, fake_rooms):
    model_config = _model_config(n_rows, fake_rooms)
    model = model_creator(model_config)

    # get_iv_array reads model.building.Re/Ce/surf_area and the physical loads,
    # which are only populated once these have run (normally done by model.setup()).
    model._build_matrices()
    model._build_loads()

    csv_path = _indoor_temperature_csv(tmp_path, n_rows, fake_rooms)
    dataset = BuildingTemperatureDataset(csv_path, sample_size=n_rows, all=True)

    start = time.perf_counter()
    iv_array = get_iv_array(model, dataset)
    elapsed = time.perf_counter() - start

    print(f"\n[get_iv_array] n_rows={n_rows:>5}  time={elapsed:7.3f}s  ({elapsed / n_rows * 1000:.3f} ms/row)")

    n_rooms = len(model.building.rooms)
    t_eval, _ = dataset.get_all_data()
    if t_eval.dim() > 1:
        t_eval = t_eval.squeeze(0)

    # Sanity check the result is actually usable, not just fast/slow.
    sample = iv_array(t_eval[len(t_eval) // 2])
    assert sample.shape[0] == n_rooms + 2
    assert torch.isfinite(sample).all()
