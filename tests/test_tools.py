import copy
import os

import numpy as np
import pytest
import torch

from rcmodel import LOAD_KEYS, InputScaling, RCModel, load_weather_csv, model_creator, write_weather_csv


@pytest.fixture
def scaling():
    rm_CA = [200, 800]  # [min, max] Capacitance/area
    C1 = [1.5 * 10**4, 10**6]
    C2 = [2.1 * 10**4, 10**5]
    R1 = [0.2, 1.2]
    R2 = [0.3, 0.9]
    R3 = [0.02, 1]
    Rin = [0.1, 1.1]
    cool = [0, 500]
    gain = [0, 100]
    solar = [0, 0.2]

    scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain, solar)
    return scaling


def test_physical_param_scaling(scaling):
    theta = 0.5 * torch.ones(scaling.get_n_params())
    theta_physical = scaling.physical_param_scaling(theta)

    assert (
        theta_physical - torch.tensor([5.0000e02, 5.0750e05, 6.0500e04, 7.0000e-01, 6.0000e-01, 5.1000e-01, 6.0000e-01])
    ).sum() < 1e-6


def test_model_param_scaling(scaling):
    model_scaled = scaling.model_param_scaling(
        torch.tensor([5.0000e02, 5.0750e05, 6.0500e04, 7.0000e-01, 6.0000e-01, 5.1000e-01, 6.0000e-01])
    )

    assert (model_scaled - (0.5 * torch.ones(scaling.get_n_params()))).sum() < 1e-6


def test_physical_loads_scaling(scaling):
    loads = torch.tensor([[0.3, 0.8], [0.1, 0.25], [0.5, 1.0]])
    cool_physical = scaling.physical_loads_scaling(loads)

    torch.testing.assert_close(cool_physical, torch.tensor([[150.0, 400.0], [10.0, 25.0], [0.1, 0.2]]))


def test_model_loads_scaling(scaling):
    loads = torch.tensor([[150.0, 400.0], [10.0, 25.0], [0.1, 0.2]])
    cool_scaled = scaling.model_loads_scaling(loads)

    torch.testing.assert_close(cool_scaled, torch.tensor([[0.3, 0.8], [0.1, 0.25], [0.5, 1.0]]))


def test_loads_scaling_needs_a_row_per_load(scaling):
    with pytest.raises(ValueError, match="one row of loads per energy range"):
        scaling.physical_loads_scaling(torch.tensor([[0.3, 0.8], [0.1, 0.25]]))


def test_solar_range_defaults_to_zero_to_one():
    """Callers written before the solar term give only cool and gain; solar then spans [0, 1]."""
    scaling = InputScaling([1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [0, 10], [0, 10])
    loads = scaling.physical_loads_scaling(torch.tensor([[0.5], [0.5], [0.25]]))
    torch.testing.assert_close(loads[2], torch.tensor([0.25]))


def expected_physical(model_config, key, n_rooms=1):
    """Physical value of a 0-1 config parameter, straight from its [min, max] range.
    A single value is broadcast to every room, a per-room array must already match."""
    lo, hi = model_config[key]
    scaled = np.broadcast_to(np.asarray(model_config["parameters"][key], dtype=float).flatten(), (n_rooms,))
    return torch.tensor(lo + scaled * (hi - lo), dtype=torch.float32)


PARAM_KEYS = ("C_rm", "C1", "C2", "R1", "R2", "R3", "Rin")  # order of Building.categorise_theta()


@pytest.mark.parametrize("load_form", ["float", "array_of_one", "array_per_room"])
def test_model_setup(get_model_config, load_form):
    model_config = get_model_config
    p = model_config["parameters"]
    n_rooms = len(model_config["room_names"])

    if load_form != "float":
        for key in PARAM_KEYS:  # np.random.rand(1) rather than a float must not break anything
            p[key] = np.random.rand(1)
    if load_form == "array_of_one":
        for key in LOAD_KEYS:
            p[key] = np.random.rand(1)
    elif load_form == "array_per_room":
        for key in LOAD_KEYS:
            p[key] = np.random.rand(n_rooms)

    model = model_creator(model_config)
    params, loads = model.get_physical_paramaters()

    expected_params = torch.cat([expected_physical(model_config, key) for key in PARAM_KEYS])
    # One row per LOAD_KEYS entry (cool, gain, solar), one column per room.
    expected_loads = torch.stack([expected_physical(model_config, key, n_rooms) for key in LOAD_KEYS])

    torch.testing.assert_close(params, expected_params, rtol=1e-4, atol=0)
    torch.testing.assert_close(loads, expected_loads, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- weather csv


@pytest.fixture
def weather_csv(tmp_path, fake_time, fake_outdoor_weather, fake_ghi):
    return write_weather_csv(tmp_path / "weather.csv", fake_time, fake_outdoor_weather, fake_ghi)


def test_weather_csv_round_trips(weather_csv, fake_time, fake_outdoor_weather, fake_ghi):
    weather = load_weather_csv(weather_csv)
    np.testing.assert_allclose(weather["time"], fake_time)
    np.testing.assert_allclose(weather["outdoor_temperature"], fake_outdoor_weather)
    np.testing.assert_allclose(weather["ghi"], fake_ghi)


def test_weather_csv_is_cached_until_the_file_changes(weather_csv, fake_time, fake_outdoor_weather):
    assert load_weather_csv(weather_csv) is load_weather_csv(weather_csv)
    with pytest.raises(ValueError):
        load_weather_csv(weather_csv)["ghi"][0] = 1.0  # shared through the cache, so read-only

    write_weather_csv(weather_csv, fake_time, fake_outdoor_weather, np.full(len(fake_time), 123.0))
    stat = os.stat(weather_csv)
    os.utime(weather_csv, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # beat coarse mtime resolution
    assert load_weather_csv(weather_csv)["ghi"][0] == 123.0


def test_weather_csv_without_ghi_is_all_zero(tmp_path, fake_time, fake_outdoor_weather):
    path = write_weather_csv(tmp_path / "no_sun.csv", fake_time, fake_outdoor_weather)
    assert not load_weather_csv(path)["ghi"].any()


@pytest.mark.parametrize(
    "ghi_fault, message",
    [(-1.0, ">= 0"), (np.nan, "NaN"), ("short", "lengths differ")],
)
def test_bad_ghi_is_rejected(tmp_path, fake_time, fake_outdoor_weather, fake_ghi, ghi_fault, message):
    ghi = np.array(fake_ghi, dtype=float)
    if ghi_fault == "short":
        ghi = ghi[:-1]
    else:
        ghi[10] = ghi_fault
    with pytest.raises(ValueError, match=message):
        write_weather_csv(tmp_path / "bad.csv", fake_time, fake_outdoor_weather, ghi)


def test_bad_ghi_in_config_arrays_is_rejected(get_model_config):
    model_config = copy.deepcopy(get_model_config)
    model_config["weather_data_ghi"] = -np.asarray(model_config["weather_data_ghi"]) - 1
    with pytest.raises(ValueError, match=">= 0"):
        model_creator(model_config)


def test_csv_route_builds_the_same_model_as_arrays(get_model_config, weather_csv, fake_time):
    from_arrays = model_creator(copy.deepcopy(get_model_config))

    csv_config = copy.deepcopy(get_model_config)
    for key in ("weather_data_outdoor_temperature", "weather_data_UTC_time", "weather_data_ghi"):
        del csv_config[key]
    csv_config["weather_csv"] = str(weather_csv)
    from_csv = model_creator(csv_config)

    t = torch.tensor(fake_time[5:200:7] + 11.0, dtype=torch.float64)  # between samples too
    np.testing.assert_allclose(from_csv.ghi(t), from_arrays.ghi(t))
    np.testing.assert_allclose(from_csv.Tout_continuous(t), from_arrays.Tout_continuous(t))


def test_missing_solar_range_is_a_clear_error(get_model_config):
    model_config = copy.deepcopy(get_model_config)
    del model_config["solar"]
    with pytest.raises(AssertionError, match="solar"):
        model_creator(model_config)


def test_missing_ghi_warns_and_means_no_solar(get_model_config, fake_time):
    model_config = copy.deepcopy(get_model_config)
    del model_config["weather_data_ghi"]
    with pytest.warns(UserWarning, match="no solar data"):
        model = model_creator(model_config)
    assert model.ghi_continuous is None
    assert not model.ghi(torch.tensor(fake_time[:10], dtype=torch.float64)).any()


def test_model_pickled_before_solar_loads_with_zero_solar(get_model_config, tmp_path):
    """load_model_path_physical must accept a model saved when loads had only cool and gain rows."""
    old = model_creator(copy.deepcopy(get_model_config))
    old.loads = torch.nn.Parameter(old.loads[:2].clone(), requires_grad=False)
    path = old.save(str(tmp_path / "old_model.pkl"))

    model_config = copy.deepcopy(get_model_config)
    model_config["parameters"] = None
    model_config["load_model_path_physical"] = path
    model = model_creator(model_config)

    assert model.loads.shape == (len(LOAD_KEYS), len(model.building.rooms))
    torch.testing.assert_close(model.loads[:2], old.loads)
    assert not model.loads[LOAD_KEYS.index("solar")].any()
    assert isinstance(RCModel.load(path), RCModel)


if __name__ == "__main__":
    pytest.main()
