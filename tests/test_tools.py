import pytest
import torch

from rcmodel import InputScaling


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


if __name__ == '__main__':
    pytest.main()
