import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from rcmodel import BuildingTemperatureDataset, RCModel, model_creator
from rcmodel.rc_model import RC_PARAM_KEYS, get_iv_array, steady_state_iv


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
    # Manually set loads to zero. Parameters are held in 0-1 machine space with no sigmoid
    # to undo, so the bottom of each range (cool=0, gain=0) is simply 0.
    model.loads = torch.nn.Parameter(torch.zeros(model.loads.shape), requires_grad=False)
    model._build_loads()

    # Set parameters for forward run:
    t_eval = torch.arange(0, 200000, 30, dtype=torch.float32)

    model.iv = 26 * torch.ones(2 + len(model.building.rooms))
    model.setup(dataset=None)

    output = model(t_eval)

    start_temp = output[0][0, 0].item()  # initial value condition

    # Both bounds need a numerical tolerance, because the trajectory converges exactly ONTO
    # the lower one. With no loads and Tout pinned at -5, the analytic equilibrium is a
    # uniform -5, but A and B are assembled in float32, so the identity
    # A @ (-5 * 1) + B @ u == 0 only holds to a ~1e-9 residual; divided through by A's small
    # diagonal that puts the model's OWN equilibrium a few microkelvin below -5 (measured:
    # -5.000002 to -5.000005 across random parameter draws). forward() now integrates
    # exactly, so it reaches that equilibrium instead of stopping short of it - the old rk4
    # path only stayed above -5 because it had not finished converging (it bottomed out
    # around -4.99994). A zero-tolerance bound therefore failed for roughly one random draw
    # in five. 1e-3 degC is three orders of magnitude below anything physically meaningful
    # and ~200x above the observed float32 artefact.
    tol = 1e-3
    assert (not torch.any(output < -5 - tol).item()) and (not torch.any(output > start_temp + tol).item()), (
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

    n_rooms = len(model.building.rooms)
    # cool and gain rows span their [0, 5000] W/m2 ranges; the solar row its default [0, 1].
    original_loads = torch.cat([3000 * torch.rand(2, n_rooms), torch.rand(1, n_rooms)])

    model.params = torch.nn.Parameter(model.scaling.model_param_scaling(original_params), requires_grad=False)
    model.loads = torch.nn.Parameter(model.scaling.model_loads_scaling(original_loads), requires_grad=False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # save model values
        path = tmpdirname + "/test_save.pkl"
        model.save(path)

        del model

        # Initialise new model and randomise parameters
        # model2 = model_n9
        # model2.init_physical()

        model2 = RCModel.load(path)

        loaded_params, loaded_loads = model2.get_physical_paramaters()

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
    for an arbitrary draw. Minimal R/C values keep the slowest time constant to ~34 minutes, so a
    24-hour dataset (~42x that) converges comfortably and still runs in a fraction of a second.

    The target is computed via the same "Tin averaged over ALL rooms, not just the ones touching
    the external wall" formula get_iv_array() uses internally (see its `Tin_agg` comment) rather
    than assuming the raw indoor temperature value - for `fake_rooms`, only 6 of 9 rooms are
    externally connected, so the true driving Tin is diluted from 21.0 down to 14.0.
    """
    model_config = get_model_config
    fake_room_names, _ = fake_rooms
    model_config["parameters"] = dict.fromkeys(
        RC_PARAM_KEYS, 0.0
    )  # scaled 0-1: minimum of every range -> smallest R*C -> fastest time constant

    dt = 30
    n_rows = 2_880  # 24 hours @ dt=30s
    t = fake_time[0] + np.arange(0, n_rows * dt, dt)
    tout = 12.0
    tin = 21.0

    model_config["weather_data_outdoor_temperature"] = np.full(n_rows, tout)
    model_config["weather_data_UTC_time"] = t
    model_config["weather_data_ghi"] = np.zeros(n_rows)

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


def _model_with_parameters(model_config, **overrides):
    """model_creator() with every parameter pinned to 0.0 (the bottom of each range) except
    the named overrides, then A/B built. 0.0 gives the smallest R*C products and therefore
    the FASTEST dynamics the ranges allow, which is deliberate: it is the hardest case for a
    fixed-step solver, so it is the one worth comparing implementations on."""
    config = dict(model_config)
    parameters = dict.fromkeys(RC_PARAM_KEYS, 0.0)
    parameters.update(overrides)
    config["parameters"] = parameters
    model = model_creator(config)
    model.setup(dataset=None)
    return model


def test_forward_matches_odeint_mid_range(get_model_config, fake_time):
    """forward() must reproduce the torchdiffeq rk4 implementation it replaced.

    forward() steps an exact first-order-hold discretisation instead of running a numerical
    ODE solver - far faster (one batched interpolation instead of one per solver sub-step,
    and a matrix exponential cached per parameter set), which is what makes a
    population-based search affordable. This pins the two together so the speedup can't
    quietly change the physics.

    Mid-range parameters put rk4 deep inside its comfort zone (dt is ~0.1% of the fastest
    time constant), so the two should agree to near machine precision. That tight tolerance
    is the real proof that these are the same method rather than two things that happen to
    look similar. A non-zero cooling load with action=1 covers the Q input path too, not
    just the free response to outdoor temperature.

    "Machine precision" here means FLOAT32 machine precision, and the limiting factor is the
    rk4 reference, not forward(). _forward_odeint() runs entirely in float32 (A/B and iv are
    float32), so over 30 steps on states of magnitude ~24 it accumulates ~1.2e-5 of pure
    round-off. Measured on this case: forward() vs an independent float64 rk4 integration of
    the same A/B agrees to 9.5e-7, which is exactly half a float32 ulp at 24 - i.e. the whole
    residual is forward()'s own float32 OUTPUT quantisation, and the two methods are
    identical to everything float64 can see. float32 rk4 vs float64 rk4 differs by 1.17e-5,
    which is what an absolute 1e-5 tolerance was actually measuring. The tolerance is
    therefore expressed relative to float32 eps rather than as an absolute temperature.
    """
    mid = dict.fromkeys(RC_PARAM_KEYS, 0.5)
    model = _model_with_parameters(get_model_config, **mid)

    t_eval = torch.tensor(fake_time[:31], dtype=torch.float64)  # 15 min at dt=30s
    start = 24 * torch.ones(2 + len(model.building.rooms))

    model.iv = start.clone()
    fast = model(t_eval, action=1).squeeze()
    model.iv = start.clone()
    reference = model._forward_odeint(t_eval, action=1).squeeze()

    assert torch.isfinite(fast).all(), "exact discretisation produced non-finite states"
    # 8 float32 ulps (~9.5e-7 relative). Observed worst case is 4.8e-7, i.e. 4 ulps.
    torch.testing.assert_close(fast, reference, rtol=8 * torch.finfo(torch.float32).eps, atol=0)


def test_forward_matches_odeint_at_the_stiffest_corner(get_model_config, fake_time):
    """The same comparison at the fastest dynamics the parameter ranges permit.

    Every parameter at its range minimum gives the smallest R*C products and so the fastest
    modes. There the 30s step is a large fraction of the fastest time constant and rk4's own
    truncation error becomes visible, so demanding agreement to 1e-5 here would really be
    demanding that the EXACT method reproduce rk4's error. The assertion is therefore
    relative to how far the trajectory actually travels: the two must agree to well within
    1% of the excursion.

    Worth knowing which way round this is - the discrepancy is rk4's, not forward()'s. The
    exact discretisation has no step-size accuracy limit at all, so the replacement is
    slightly MORE accurate exactly where the old one was weakest, which is the corner of the
    space a random search spends real time in.
    """
    model = _model_with_parameters(get_model_config, cool=0.5, gain=0.2)

    t_eval = torch.tensor(fake_time[:31], dtype=torch.float64)
    start = 24 * torch.ones(2 + len(model.building.rooms))

    model.iv = start.clone()
    fast = model(t_eval, action=1).squeeze()
    model.iv = start.clone()
    reference = model._forward_odeint(t_eval, action=1).squeeze()

    assert torch.isfinite(fast).all(), "exact discretisation produced non-finite states"

    excursion = (reference.max() - reference.min()).abs().item()
    assert excursion > 1e-2, "trajectory barely moved, so agreement would prove nothing"

    difference = (fast - reference).abs().max().item()
    assert difference < 0.01 * excursion, f"max difference {difference:.2e} exceeds 1% of the {excursion:.2e} excursion"


def test_forward_is_repeatable(get_model_config, fake_time):
    """Two identical calls must give identical output.

    The discretisation is cached across calls (keyed by dt, cleared by setup()), so this
    would catch the cache being mutated or state leaking between calls - which would show up
    in a PBT run as trials that score differently on re-evaluation for no reason.
    """
    model = _model_with_parameters(get_model_config, cool=0.5)
    t_eval = torch.tensor(fake_time[:31], dtype=torch.float64)
    start = 24 * torch.ones(2 + len(model.building.rooms))

    model.iv = start.clone()
    first = model(t_eval, action=1)
    model.iv = start.clone()
    second = model(t_eval, action=1)

    assert torch.equal(first, second)


def test_forward_responds_to_action(get_model_config, fake_time):
    """Cooling on must end colder than cooling off - otherwise the action is being dropped
    somewhere between the environment and the input vector, and every policy would score the
    same."""
    model = _model_with_parameters(get_model_config, cool=1.0)
    t_eval = torch.tensor(fake_time[:31], dtype=torch.float64)
    start = 24 * torch.ones(2 + len(model.building.rooms))

    model.iv = start.clone()
    cooling_off = model(t_eval, action=0).squeeze()
    model.iv = start.clone()
    cooling_on = model(t_eval, action=1).squeeze()

    assert (cooling_on[-1, 2:] < cooling_off[-1, 2:]).all()


def _run(model, t_eval, action=0, start=24.0):
    model.iv = start * torch.ones(2 + len(model.building.rooms))
    return model(t_eval, action=action).squeeze()


def test_forward_matches_odeint_with_solar(get_model_config, fake_time):
    """With a time-varying GHI the heat input is no longer constant across a step, so this is
    the case that checks forward()'s first-order hold on Q against the rk4 reference, which
    queries GHI at every solver sub-step. The window is the start of the synthetic day, where
    GHI rises fastest, and solar is at the top of its range so it dominates the heat input."""
    mid = dict.fromkeys(RC_PARAM_KEYS, 0.5)
    mid["solar"] = 1.0
    model = _model_with_parameters(get_model_config, **mid)
    t_eval = torch.tensor(fake_time[:31], dtype=torch.float64)

    ghi = model.ghi(t_eval)
    assert ghi[-1] - ghi[0] > 50, "GHI barely changes over the window, so this would not test a varying input"

    fast = _run(model, t_eval, action=1)
    model.iv = 24 * torch.ones(2 + len(model.building.rooms))
    reference = model._forward_odeint(t_eval, action=1).squeeze()

    torch.testing.assert_close(fast, reference, rtol=8 * torch.finfo(torch.float32).eps, atol=0)

    no_solar = _model_with_parameters(get_model_config, **{**mid, "solar": 0.0})
    effect = (fast - _run(no_solar, t_eval, action=1)).abs().max().item()
    assert effect > 1e-3, "solar made no visible difference, so the agreement above proves nothing about it"


def test_zero_solar_is_the_same_as_no_solar_data(get_model_config, fake_time):
    """p = 0 with GHI data, and any p without GHI data, must both give exactly the model as it
    was before the solar term existed - a regression guard for every run without solar data."""
    t_eval = torch.tensor(fake_time[:31], dtype=torch.float64)
    zero_p = _model_with_parameters(get_model_config, cool=0.5, gain=0.5, solar=0.0)
    no_data = _model_with_parameters(get_model_config, cool=0.5, gain=0.5, solar=1.0)
    no_data.ghi_continuous = None

    torch.testing.assert_close(_run(zero_p, t_eval, action=1), _run(no_data, t_eval, action=1), rtol=0, atol=0)


def test_model_pickled_before_solar_still_runs(get_model_config, fake_time):
    """Models pickled before the solar term have no ghi_continuous attribute at all."""
    model = _model_with_parameters(get_model_config, gain=0.5)
    del model.ghi_continuous
    np.testing.assert_array_equal(model.ghi(torch.tensor(fake_time[:5], dtype=torch.float64)), np.zeros(5))
    assert torch.isfinite(_run(model, torch.tensor(fake_time[:31], dtype=torch.float64))).all()


def test_solar_gain_is_p_times_ghi_times_floor_area(get_model_config, fake_time):
    """Pins the units: under a constant GHI of G W/m2, solar fraction p must heat the rooms
    exactly as a constant gain of p * G W/m2 of floor area does."""
    model_config = dict(get_model_config)
    ghi_value = 40.0
    model_config["weather_data_ghi"] = np.full(len(fake_time), ghi_value)

    low, high = model_config["solar"]
    p_scaled = 0.5
    p = low + p_scaled * (high - low)
    gain_low, gain_high = model_config["gain"]
    gain_scaled = (p * ghi_value - gain_low) / (gain_high - gain_low)
    assert 0 < gain_scaled < 1

    t_eval = torch.tensor(fake_time[:121], dtype=torch.float64)
    via_solar = _run(_model_with_parameters(model_config, solar=p_scaled), t_eval)
    via_gain = _run(_model_with_parameters(model_config, gain=gain_scaled), t_eval)

    assert (via_solar[-1, 2:] - via_solar[0, 2:]).abs().max() > 1e-3, "no heating to compare"
    torch.testing.assert_close(via_solar, via_gain, rtol=1e-6, atol=1e-5)


def test_solar_response_is_linear_in_p(get_model_config, fake_time):
    """The system is linear, so the change solar makes must double when p doubles."""
    t_eval = torch.tensor(fake_time[:121], dtype=torch.float64)
    baseline = _run(_model_with_parameters(get_model_config, solar=0.0), t_eval).double()
    single = _run(_model_with_parameters(get_model_config, solar=0.25), t_eval).double() - baseline
    double = _run(_model_with_parameters(get_model_config, solar=0.5), t_eval).double() - baseline

    assert single.abs().max() > 1e-2
    torch.testing.assert_close(double, 2 * single, rtol=1e-3, atol=1e-4)


def test_slowest_time_constant_orders_parameter_sets(get_model_config):
    """The plausibility filter leans on this: big R*C products must report as slower.

    The parameter ranges permit time constants of hundreds of days, which cannot be
    identified from a few weeks of data. pbt.RCPolicyTrainable uses this to sideline those
    draws instead of spending a population slot on them.
    """
    fastest = _model_with_parameters(get_model_config)  # every parameter at range minimum
    slowest = _model_with_parameters(get_model_config, **dict.fromkeys(RC_PARAM_KEYS, 1.0))

    tau_fast = fastest.slowest_time_constant()
    tau_slow = slowest.slowest_time_constant()

    assert tau_fast > 0, "a stable thermal system has a positive time constant"
    assert tau_slow > tau_fast
    # Sanity-check the magnitude rather than an exact value: the minimum-parameter draw
    # settles in well under a day, the maximum-parameter one takes far longer than a week.
    assert tau_fast < 24 * 60**2
    assert tau_slow > 7 * 24 * 60**2


if __name__ == "__main__":
    pytest.main()
