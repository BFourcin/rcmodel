import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from rcmodel import BuildingTemperatureDataset, RCModel, model_creator
from rcmodel.rc_model import get_iv_array, steady_state_iv


@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("model_n2"),
        pytest.lazy_fixture("model_n9"),
    ],
)
def test_run_model(model):
    """
    Check if the model is physically valid by testing if it violates the laws of thermodynamics. Specifically, the function
    ensures that the model does not gain temperature beyond its starting condition and that it does not lose temperature so
    that it is colder than the outside temperature, which is physically impossible.

    Args:
        model (torch.nn.Module): The PyTorch model to test.

    Raises:
        AssertionError: If the model gains or loses energy from the system.
    """
    # Manually set loads to zero:
    loads = torch.logit(torch.zeros(model.loads.shape))
    model.loads = torch.nn.Parameter(loads)
    model._build_loads()

    # Set parameters for forward run:
    t_eval = torch.arange(0, 200000, 30, dtype=torch.float32)

    model.iv = 26 * torch.ones(2 + len(model.building.rooms))
    model.setup(dataset=None)

    output = model(t_eval)

    start_temp = output[0][0, 0].item()  # initial value condition

    assert (not torch.any(output < -5).item()) and (not torch.any(output > start_temp).item()), (
        "Model has gained or lost energy from the system"
    )


def test_save_load(model_n9):
    """
    Test if model parameters are unchanged when saved and loaded with different scaling limits.

    This test creates a new `RCModel` instance, sets its parameters and loads to some values, saves it to a temporary file,
    loads the saved model into a new instance, and checks if the loaded parameters and loads match the original values.
    If the loaded parameters or loads differ from the original values by more than 1e-3, the test fails.

    Args:
        model_n9 (RCModel): An instance of the `RCModel` class to be tested.

    Raises:
        AssertionError: If the loaded parameters or loads differ from the original values by more than 1e-3.
    """
    model = model_n9

    rm_cap = 500
    ex_cap = [1.6e4, 2e4]
    ex_r = [0.2, 0.5, 1.2]
    wl_r = 0.8

    original_params = torch.tensor([rm_cap, ex_cap[0], ex_cap[1], ex_r[0], ex_r[1], ex_r[2], wl_r])

    original_loads = 3000 * torch.rand(2, len(model.building.rooms))

    model.params = torch.nn.Parameter(torch.logit(model.scaling.model_param_scaling(original_params)))
    model.loads = torch.nn.Parameter(torch.logit(model.scaling.model_loads_scaling(original_loads)))

    with tempfile.TemporaryDirectory() as tmpdirname:
        # save model values
        path = tmpdirname + "/test_save.pkl"
        model.save(path)

        del model

        # Initialise new model and randomise parameters
        # model2 = model_n9
        # model2.init_physical()

        model2 = RCModel.load(path)

        loaded_params = model2.scaling.physical_param_scaling(model2.transform(model2.params))
        loaded_loads = model2.scaling.physical_loads_scaling(model2.transform(model2.loads))

        diff_params = abs(loaded_params - original_params)
        diff_loads = abs(loaded_loads - original_loads)

    assert (diff_params < 1e-3).all() and (diff_loads < 1e-3).all(), "model parameters are changing during a save and load"


def test_get_iv_array_converges_to_steady_state(get_model_config, fake_rooms, fake_time, tmp_path):
    """With near-constant outdoor/indoor temperatures held for long enough, get_iv_array()'s
    latent node estimates should settle close to the analytic steady-state solution
    (steady_state_iv()) - an implementation-agnostic invariant, independent of how the
    integration itself is done.

    Overrides get_model_config's (random) physical parameters with fixed, minimal ones instead of
    using its random seed-42 draw directly: the physical parameter ranges legally permit R*C time
    constants of hundreds of days (confirmed empirically - the raw seed-42 draw has a ~439-day
    slow mode), so no dataset length short enough to run as a fast test would actually converge
    for an arbitrary draw. Minimal R/C values keep the slowest time constant to ~72 minutes, so a
    24-hour dataset (~20x that) converges comfortably and still runs in a fraction of a second.

    The target is computed via the same "Tin averaged over ALL rooms, not just the ones touching
    the external wall" formula get_iv_array() uses internally (see its `Tin_agg` comment) rather
    than assuming the raw indoor temperature value - for `fake_rooms`, only 6 of 9 rooms are
    externally connected, so the true driving Tin is diluted from 21.0 down to 14.0.
    """
    model_config = get_model_config
    fake_room_names, _ = fake_rooms
    model_config["parameters"] = dict.fromkeys(
        ["C_rm", "C1", "C2", "R1", "R2", "R3", "Rin", "cool", "gain"], 0.0
    )  # scaled 0-1: minimum of every range -> smallest R*C -> fastest time constant

    dt = 30
    n_rows = 2_880  # 24 hours @ dt=30s
    t = fake_time[0] + np.arange(0, n_rows * dt, dt)
    tout = 12.0
    tin = 21.0

    model_config["weather_data_outdoor_temperature"] = np.full(n_rows, tout)
    model_config["weather_data_UTC_time"] = t

    indoor_temps = tin + np.random.default_rng(3).uniform(-1e-3, 1e-3, size=(n_rows, len(fake_room_names)))
    df = pd.DataFrame(indoor_temps, columns=fake_room_names)
    df.insert(0, "time", t)
    df.insert(0, "date-time", pd.to_datetime(t, unit="s"))
    csv_path = tmp_path / "steady_state_indoor_temperature.csv"
    df.to_csv(csv_path, index=False)

    model = model_creator(model_config)
    model._build_matrices()
    model._build_loads()
    dataset = BuildingTemperatureDataset(csv_path, sample_size=n_rows, all=True)

    iv_array = get_iv_array(model, dataset)

    # Query with t_eval read back from the dataset (not the `t` array above directly) - writing
    # epoch-scale floats to CSV and reading them back can shift the last value by enough to fall
    # just outside iv_array's interpolation domain otherwise.
    t_eval, _ = dataset.get_all_data()
    if t_eval.dim() > 1:
        t_eval = t_eval.squeeze(0)

    external_rooms = model.building.connectivity_matrix[0, 1:]
    tin_agg = (torch.full((len(fake_room_names),), tin) * external_rooms).float().mean().item()

    steady = steady_state_iv(model, torch.tensor(tout), torch.tensor(tin_agg)).squeeze()
    final_state = iv_array(t_eval[-1])

    assert torch.allclose(final_state[0:2], steady[0:2], atol=0.05), (
        f"latent state {final_state[0:2]} did not converge to steady state {steady[0:2]}"
    )


def test_get_iv_array_handles_realistic_timestamps(get_model_config, full_building_dataset):
    """Regression test for a real bug: get_iv_array() internally shifted its time array to a
    relative (starts-at-zero) origin, then queried model.Tout_continuous - whose domain is the
    dataset's own absolute (epoch-scale) time - with that shifted array. On any dataset that
    doesn't already start at t=0 (i.e. every real dataset), that queried far outside
    Tout_continuous's domain, and Interp1D's default extrapolation silently returned NaN for
    every row, propagating into an all-NaN iv_array - this is what broke a real training run.

    conftest's fixtures are realistic-epoch by construction (see T_ORIGIN in conftest.py), so
    this would also be caught by the steady-state test above - this test exists to name and pin
    down the exact failure mode directly.
    """
    model = model_creator(get_model_config)
    model._build_matrices()
    model._build_loads()

    iv_array = get_iv_array(model, full_building_dataset)

    t_eval, _ = full_building_dataset.get_all_data()
    if t_eval.dim() > 1:
        t_eval = t_eval.squeeze(0)

    assert t_eval[0].item() > 1e9, "fixture should use realistic epoch-scale absolute times"

    for idx in [0, len(t_eval) // 2, len(t_eval) - 1]:
        sample = iv_array(t_eval[idx])
        assert torch.isfinite(sample).all(), f"idx={idx}: iv_array returned non-finite values: {sample}"


if __name__ == "__main__":
    pytest.main()
