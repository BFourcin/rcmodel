import torch
import pytest

from rcmodel.tools import InputScaling

@pytest.fixture
def scaling():
    rm_CA = [200, 800] #[min, max] Capacitance/area
    ex_C = [1.5*10**4, 10**6] #Capacitance
    R = [0.2, 1.2] # Resistance ((K.m^2)/W)
    Q_avg = [0, 100] # Heat/area (W/m^2)

    scaling = InputScaling(rm_CA, ex_C, R, Q_avg)
    return scaling


def test_physical_scaling(scaling):

    scale_f = scaling.physical_scaling

    theta = 0.5 * torch.ones(scaling.get_n_params())
    theta_physical = scale_f(theta)

    assert torch.equal(theta_physical, torch.tensor([5.0000e+02, 5.0750e+05, 5.0750e+05, 7.0000e-01, 7.0000e-01, 7.0000e-01, 7.0000e-01, 5.0000e+01]))

def test_model_scaling(scaling):

    model_scaled = scaling.model_scaling(torch.tensor([5.0000e+02, 5.0750e+05, 5.0750e+05, 7.0000e-01, 7.0000e-01, 7.0000e-01, 7.0000e-01, 5.0000e+01]))

    assert torch.equal(model_scaled, 0.5*torch.ones(scaling.get_n_params()))
