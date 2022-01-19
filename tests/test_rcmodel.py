import pytest
import torch
import tempfile
import numpy as np

from rcmodel.building import Building
from rcmodel.RCModel import RCModel
from rcmodel.tools import InputScaling
from rcmodel.optimisation import OptimiseRC


# Function for constant outside temperature. try/except allows for broadcasting of value.
def dummy_tout(t):
    try:
        return -5 * torch.ones(len(t))

    except TypeError:
        return torch.tensor(-5)


@pytest.mark.parametrize(
    'building',
    [
        pytest.lazy_fixture('building_n2'),
        pytest.lazy_fixture('building_n9'),
    ], )
def test_run_model(building):

    # Initialise scaling methods:
    rm_CA = [200, 800]  # [min, max] Capacitance/area
    ex_C = [1.5 * 10 ** 4, 10 ** 6]  # Capacitance
    R = [0.2, 1.2]  # Resistance ((K.m^2)/W)
    Q_limit = [0, 0]  # gain limit (W/m^2) no additional energy

    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)

    # Initialise RCModel with the building and InputScaling
    transform = torch.sigmoid
    model = RCModel(building, scaling, dummy_tout, transform)
    model.cooling = torch.nn.Parameter(model.cooling * 0)  # no heating

    # Set parameters for forward run:
    t_eval = torch.arange(0, 200000, 30, dtype=torch.float32)

    output = model(t_eval)

    start_temp = output[0][0, 0].item()  # initial value condition

    assert (not torch.any(output < -5).item()) and (not torch.any(output > start_temp).item()),\
        "Model has gained or lost energy from the system"


def test_save_load(building_n9):
    """
    tests if parameters are unchanged when saved and loaded with different scaling limits.
    """

    # Initialise scaling methods:
    rm_CA = [200, 800]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e6]  # Capacitance
    R = [0.1, 1.3]  # Resistance ((K.m^2)/W)
    Q_limit = [-10000, 10000]  # gain limit (W/m^2)

    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
    transform = torch.sigmoid

    model = RCModel(building_n9, scaling, dummy_tout, transform)

    rm_cap = 500
    ex_cap = [1e4, 2e4]
    ex_r = [0.2, 0.5, 1.2]
    wl_r = 0.8
    gain = 2400

    original_params = torch.tensor([rm_cap, ex_cap[0], ex_cap[1], ex_r[0], ex_r[1], ex_r[2], wl_r, gain])
    original_cooling = 3000 * torch.rand(len(model.building.rooms))

    model.params = torch.nn.Parameter(torch.logit(model.scaling.model_param_scaling(original_params)))
    model.cooling = torch.nn.Parameter(torch.logit(model.scaling.model_cooling_scaling(original_cooling)))

    with tempfile.TemporaryDirectory() as tmpdirname:
        num = 0
        # save model values
        model.save(model_id=num, dir_path=tmpdirname)

        del model

        # Initialise new model with different scaling:
        rm_CA = [50, 901]
        ex_C = [1e2, 1e5]
        R = [0.2, 2]
        Q_limit = [-5200, 8100]

        scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
        transform = torch.sigmoid

        model2 = RCModel(building_n9, scaling, dummy_tout, transform)

        path = f"{tmpdirname}/rcmodel{num}.pt"
        model2.load(path=path)

        loaded_params = model2.scaling.physical_param_scaling(transform(model2.params))
        loaded_cooling = model2.scaling.physical_cooling_scaling(transform(model2.cooling))

        diff_params = abs(loaded_params-original_params)
        diff_cooling = abs(loaded_cooling-original_cooling)

    assert (diff_params < 1e-3).all() and (diff_cooling < 1e-3).all(), 'model parameters are changing during a save and load'


if __name__ == '__main__':
    pytest.main()

    print("__main__ reached")
