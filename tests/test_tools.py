import numpy as np
import pytest
import torch

from rcmodel import InputScaling, model_creator


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

    scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain)
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
    loads = torch.tensor([[0.3, 0.8], [0.1, 0.25]])
    cool_physical = scaling.physical_loads_scaling(loads)

    assert torch.equal(cool_physical, torch.tensor([[150.0, 400.0], [10.0, 25.0]]))


def test_model_loads_scaling(scaling):
    loads = torch.tensor([[150.0, 400.0], [10.0, 25.0]])
    cool_scaled = scaling.model_loads_scaling(loads)

    assert torch.equal(cool_scaled, torch.tensor([[0.3, 0.8], [0.1, 0.25]]))


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
        p["cool"], p["gain"] = np.random.rand(1), np.random.rand(1)
    elif load_form == "array_per_room":
        p["cool"], p["gain"] = np.random.rand(n_rooms), np.random.rand(n_rooms)

    model = model_creator(model_config)
    params, loads = model.get_physical_paramaters()

    expected_params = torch.cat([expected_physical(model_config, key) for key in PARAM_KEYS])
    # Row 0 is cool, row 1 is gain, one column per room.
    expected_loads = torch.stack(
        [expected_physical(model_config, "cool", n_rooms), expected_physical(model_config, "gain", n_rooms)]
    )

    torch.testing.assert_close(params, expected_params, rtol=1e-4, atol=0)
    torch.testing.assert_close(loads, expected_loads, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main()
