"""
Correctness checks for the FOH-based `get_iv_array()` (src/rcmodel/rc_model.py).

`get_iv_array()` used to numerically integrate the latent 2-node subsystem with
torchdiffeq's RK4 solver, calling `Interp1D` per RK4 sub-step - see
tests/test_iv_array_performance.py for the profiling that motivated replacing it.
Since Tout/Tin are exogenous inputs (not functions of state) and the subsystem is
linear time-invariant, it's now discretized exactly (first-order-hold) instead.

`_get_iv_array_reference_rk4` below is a frozen copy of the *old* RK4-based
implementation, kept only so the new implementation's output can be checked
against it - it is not used anywhere else.

All time fixtures here use a realistic, non-zero, epoch-scale time origin
(T_ORIGIN below) rather than starting at t=0. This matters: get_iv_array() had a
real bug where it queried model.Tout_continuous (whose domain is the dataset's
own absolute/epoch time) using a relative time array that had been shifted to
start at 0, which silently returned NaN via Interp1D's out-of-domain
extrapolation on any real (non-zero-origin) dataset. A fixture that starts at
t=0 makes that shift a no-op and completely hides the bug - which is exactly
what let it ship - so every fixture here deliberately avoids starting at zero.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D

from rcmodel import BuildingTemperatureDataset, model_creator
from rcmodel.rc_model import get_iv_array, steady_state_iv

DT = 30  # seconds, matches production sampling rate
T_ORIGIN = 1_600_000_000  # realistic unix-epoch-scale start time (2020-09-13), deliberately not 0


def _model_config(n_rows, fake_rooms):
    np.random.seed(42)
    fake_room_names, fake_room_coordinates = fake_rooms
    n_seconds = n_rows * DT
    t = T_ORIGIN + np.arange(0, n_seconds, DT)
    outdoor = 10 + 5 * np.sin(2 * np.pi * np.arange(0, n_seconds, DT) / n_seconds)

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


def _indoor_temperature_csv(tmp_path, n_rows, fake_rooms, constant=False):
    fake_room_names, _ = fake_rooms
    n_seconds = n_rows * DT
    t_rel = np.arange(0, n_seconds, DT)
    t = T_ORIGIN + t_rel

    if constant:
        # Steady-state scenario: (near-)constant indoor temperature.
        rng = np.random.default_rng(3)
        indoor_temps = 21.0 + rng.uniform(-1e-3, 1e-3, size=(n_rows, len(fake_room_names)))
    else:
        rng = np.random.default_rng(7)
        base_indoor = 22 + 2 * np.sin(2 * np.pi * (t_rel - 3 * 60**2) / n_seconds)
        room_offsets = rng.uniform(-1.0, 1.0, size=len(fake_room_names))
        indoor_temps = base_indoor[:, None] + room_offsets[None, :]

    df = pd.DataFrame(indoor_temps, columns=fake_room_names)
    df.insert(0, "time", t)
    df.insert(0, "date-time", pd.to_datetime(t, unit="s"))

    csv_path = tmp_path / f"indoor_temperature_{n_rows}_{constant}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _get_iv_array_reference_rk4(model, dataset):
    """Frozen copy of the pre-optimization RK4/odeint implementation, for comparison only."""
    with torch.no_grad():
        t_eval, temp_data = dataset.get_all_data()
        Tin_continuous = Interp1D(t_eval, temp_data[:, 0 : len(model.building.rooms)].T, method="linear")

        bl = model.building

        A = torch.zeros([2, 2])
        A[0, 0] = bl.surf_area * (-1 / (bl.Re[0] * bl.Ce[0]) - 1 / (bl.Re[1] * bl.Ce[0]))
        A[0, 1] = bl.surf_area / (bl.Re[1] * bl.Ce[0])
        A[1, 0] = bl.surf_area / (bl.Re[1] * bl.Ce[1])
        A[1, 1] = bl.surf_area * (-1 / (bl.Re[1] * bl.Ce[1]) - 1 / (bl.Re[2] * bl.Ce[1]))

        B = torch.zeros([2, 2])
        B[0, 0] = bl.surf_area / (bl.Re[0] * bl.Ce[0])
        B[1, 1] = bl.surf_area / (bl.Re[2] * bl.Ce[1])

        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        avg_tout = model.Tout_continuous(t_eval).mean()
        avg_tin = Tin_continuous(t_eval).mean()

        t0 = t_eval[0]
        t_eval = t_eval - t0

        model.iv = steady_state_iv(model, avg_tout, avg_tin)

        def latent_f_ode(t, x):
            Tout = model.Tout_continuous(t.item() + t0)
            external_rooms = bl.connectivity_matrix[0, 1:]
            Tin = (Tin_continuous(t.item() + t0) * external_rooms).mean()
            u = torch.tensor([[Tout], [Tin]])
            return A @ x + B @ u.to(torch.float32)

        integrate = odeint(latent_f_ode, model.iv[0:2], t_eval, method="rk4")
        integrate = integrate.squeeze()

        iv_array = torch.empty(len(integrate), len(bl.rooms) + 2)
        iv_array[:, 0:2] = integrate
        iv_array[:, 2:] = Tin_continuous(t_eval + t0).T
        iv_array = Interp1D(t_eval + t0, iv_array.T, method="linear")

    return iv_array


@pytest.mark.parametrize("n_rows", [360, 1440])
def test_get_iv_array_matches_reference_rk4(n_rows, tmp_path, fake_rooms):
    """The FOH-based implementation should closely reproduce what the old
    RK4+linear-interpolation implementation was already approximating."""
    model_config = _model_config(n_rows, fake_rooms)
    csv_path = _indoor_temperature_csv(tmp_path, n_rows, fake_rooms)

    model_new = model_creator(model_config)
    model_new._build_matrices()
    model_new._build_loads()
    dataset_new = BuildingTemperatureDataset(csv_path, sample_size=n_rows, all=True)
    iv_new = get_iv_array(model_new, dataset_new)

    model_ref = model_creator(model_config)
    model_ref._build_matrices()
    model_ref._build_loads()
    dataset_ref = BuildingTemperatureDataset(csv_path, sample_size=n_rows, all=True)
    iv_ref = _get_iv_array_reference_rk4(model_ref, dataset_ref)

    t_eval, _ = dataset_new.get_all_data()
    if t_eval.dim() > 1:
        t_eval = t_eval.squeeze(0)

    for idx in [0, n_rows // 4, n_rows // 2, 3 * n_rows // 4, n_rows - 1]:
        tq = t_eval[idx]
        diff = (iv_new(tq) - iv_ref(tq)).abs().max().item()
        assert diff < 0.01, f"idx={idx}: new vs reference RK4 differ by {diff} degrees"


def test_get_iv_array_converges_to_steady_state(tmp_path, fake_rooms):
    """With near-constant outdoor/indoor temperatures, the latent node estimates
    should settle close to the analytic steady-state solution - an
    implementation-agnostic invariant, independent of how the integration is done."""
    n_rows = 2880  # 1 day @ dt=30s, several thermal time constants for this toy building
    model_config = _model_config(n_rows, fake_rooms)
    model_config["weather_data_outdoor_temperature"] = np.full(n_rows, 12.0)
    csv_path = _indoor_temperature_csv(tmp_path, n_rows, fake_rooms, constant=True)

    model = model_creator(model_config)
    model._build_matrices()
    model._build_loads()
    dataset = BuildingTemperatureDataset(csv_path, sample_size=n_rows, all=True)

    iv_array = get_iv_array(model, dataset)

    t_eval, _ = dataset.get_all_data()
    if t_eval.dim() > 1:
        t_eval = t_eval.squeeze(0)

    steady = steady_state_iv(model, torch.tensor(12.0), torch.tensor(21.0)).squeeze()
    final_state = iv_array(t_eval[-1])

    assert torch.allclose(final_state[0:2], steady[0:2], atol=0.05), (
        f"latent state {final_state[0:2]} did not converge to steady state {steady[0:2]}"
    )


def test_get_iv_array_handles_large_absolute_timestamps(tmp_path, fake_rooms):
    """Regression test for a real bug: get_iv_array() internally shifted its time
    array to a relative (starts-at-zero) origin for the integration step, but then
    queried model.Tout_continuous - whose domain is the dataset's own absolute
    (epoch-scale) time - with that shifted array. On any dataset that doesn't
    already start at t=0 (i.e. every real dataset), that queried far outside
    Tout_continuous's domain, and Interp1D's default extrapolation silently
    returned NaN for every row, propagating into an all-NaN iv_array.

    T_ORIGIN in this module's fixtures is already non-zero, so this would also be
    caught by the other tests above - this test exists to name and pin down the
    exact failure mode directly.
    """
    n_rows = 720
    model_config = _model_config(n_rows, fake_rooms)
    csv_path = _indoor_temperature_csv(tmp_path, n_rows, fake_rooms)

    model = model_creator(model_config)
    model._build_matrices()
    model._build_loads()
    dataset = BuildingTemperatureDataset(csv_path, sample_size=n_rows, all=True)

    iv_array = get_iv_array(model, dataset)

    t_eval, _ = dataset.get_all_data()
    if t_eval.dim() > 1:
        t_eval = t_eval.squeeze(0)

    assert t_eval[0].item() > 1e9, "test fixture should use realistic epoch-scale absolute times"

    for idx in [0, n_rows // 2, n_rows - 1]:
        sample = iv_array(t_eval[idx])
        assert torch.isfinite(sample).all(), f"idx={idx}: iv_array returned non-finite values: {sample}"
