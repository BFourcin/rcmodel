import pytest
import torch

from rcmodel.tools import InputScaling


@pytest.fixture
def scaling():
    rm_CA = [200, 800]  # [min, max] Capacitance/area
    ex_C = [1.5 * 10 ** 4, 10 ** 6]  # Capacitance
    R = [0.2, 1.2]  # Resistance ((K.m^2)/W)
    Q_limit = [-300, 500]  # gain limit (W/m^2)

    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
    return scaling


def test_physical_param_scaling(scaling):
    theta = 0.5 * torch.ones(scaling.get_n_params())
    theta_physical = scaling.physical_param_scaling(theta)
    print(theta_physical)
    assert torch.equal(theta_physical, torch.tensor(
        [5.0000e+02, 5.0750e+05, 5.0750e+05, 7.0000e-01, 7.0000e-01, 7.0000e-01, 7.0000e-01, 1.0000e+02]))


def test_model_param_scaling(scaling):
    model_scaled = scaling.model_param_scaling(
        torch.tensor([5.0000e+02, 5.0750e+05, 5.0750e+05, 7.0000e-01, 7.0000e-01, 7.0000e-01, 7.0000e-01, 1.0000e+02]))

    assert torch.equal(model_scaled, 0.5 * torch.ones(scaling.get_n_params()))


def test_physical_cooling_scaling(scaling):
    cool_physical = scaling.physical_cooling_scaling(0.2)

    assert cool_physical == 100


def test_model_cooling_scaling(scaling):
    cool_scaled = scaling.model_cooling_scaling(100)

    assert cool_scaled == 0.2


if __name__ == '__main__':
    pytest.main()
