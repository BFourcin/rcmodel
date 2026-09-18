import pytest
import torch
import numpy as np

from rcmodel import InputScaling
from rcmodel import model_creator


@pytest.fixture
def scaling():
    rm_CA = [200, 800]  # [min, max] Capacitance/area
    C1 = [1.5 * 10 ** 4, 10 ** 6]
    C2 = [2.1 * 10 ** 4, 10 ** 5]
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

    assert (theta_physical - torch.tensor(
        [5.0000e+02, 5.0750e+05, 6.0500e+04, 7.0000e-01, 6.0000e-01, 5.1000e-01, 6.0000e-01])).sum() < 1e-6


def test_model_param_scaling(scaling):
    model_scaled = scaling.model_param_scaling(
        torch.tensor([5.0000e+02, 5.0750e+05, 6.0500e+04, 7.0000e-01, 6.0000e-01, 5.1000e-01, 6.0000e-01]))

    assert (model_scaled - (0.5 * torch.ones(scaling.get_n_params()))).sum() < 1e-6


def test_physical_loads_scaling(scaling):
    loads = torch.tensor([[0.3, 0.8], [0.1, 0.25]])
    cool_physical = scaling.physical_loads_scaling(loads)

    assert torch.equal(cool_physical,  torch.tensor([[150., 400.], [10., 25.]]))


def test_model_loads_scaling(scaling):
    loads = torch.tensor([[150., 400.], [10., 25.]])
    cool_scaled = scaling.model_loads_scaling(loads)

    assert torch.equal(cool_scaled, torch.tensor([[0.3, 0.8], [0.1, 0.25]]))


def test_model_setup():
    np.random.seed(42)
    n = 24*60**2
    fake_time = np.arange(0, n, 30)
    fake_weather = 10 + 5 * np.sin(2 * np.pi * fake_time / n)

    model_config = {
        # Ranges:
        "C_rm": [1e3, 1e5],  # [min, max] Capacitance/m2
        "C1": [1e5, 1e8],  # Capacitance
        "C2": [1e5, 1e8],
        "R1": [0.1, 5],  # Resistance ((K.m^2)/W)
        "R2": [0.1, 5],
        "R3": [0.5, 6],
        "Rin": [0.1, 5],
        "cool": [0, 50],  # Cooling limit in W/m2
        "gain": [0, 5],  # Gain limit in W/m2
        "room_names": ["rm1", "rm2", "rm3", "rm4", "rm5", "rm6", "rm7", "rm8", "rm9"],
        "room_coordinates": [
            [[0, 0], [5, 0], [5, 5], [0, 5]],
            [[5, 0], [10, 0], [10, 2], [10, 4], [10, 5], [5, 5]],
            [[0, 5], [5, 5], [10, 5], [10, 6], [10, 8], [10, 10]],
            [[10, 10], [12, 10], [12, 8], [10, 8]],
            [[10, 8], [12, 8], [12, 6], [10, 6]],
            [[10, 6], [12, 6], [12, 4], [10, 4], [10, 5]],
            [[10, 4], [12, 4], [12, 2], [10, 2]],
            [[10, 2], [12, 2], [12, 0], [10, 0]],
            [[12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [14, 10], [14, 0]],
        ],
        "weather_data_outdoor_temperature": fake_weather,
        "weather_data_UTC_time": fake_time,
        "cooling_policy": None,
        "load_model_path_policy": None,  # './prior_policy.pt',  # or None
        "load_model_path_physical": None,  # or None
        "parameters": {
            "C_rm": np.random.rand(1).item(),
            "C1": np.random.rand(1).item(),
            "C2": np.random.rand(1).item(),
            "R1": np.random.rand(1).item(),
            "R2": np.random.rand(1).item(),
            "R3": np.random.rand(1).item(),
            "Rin": np.random.rand(1).item(),
            "cool": np.random.rand(1).item(),  # 0.09133423646610082
            "gain": np.random.rand(1).item(),  # 0.9086668150306394
        }

    }

    model = model_creator(model_config)

    params, loads = model.get_physical_paramaters()
    scaled_parameters_from_model = torch.cat((params, loads.flatten()))

    parameters_from_config = []
    for p in model_config['parameters']:
        parameters_from_config.append(model_config['parameters'][p])

    config_params = model.scaling.physical_param_scaling(parameters_from_config[0:-2])
    config_loads = model.scaling.physical_loads_scaling(torch.tensor(parameters_from_config[-2:]).reshape(2, 1))

    scaled_parameters_from_config = torch.cat((config_params, config_loads.flatten()))

    assert (abs(scaled_parameters_from_config - scaled_parameters_from_model) < 1e-2).all(),\
        "Parameters from the model are not matching with parameters provided in the config."


if __name__ == '__main__':
    pytest.main()
