import pytest
import torch
import tempfile
import numpy as np

from rcmodel import Building
from rcmodel import RCModel
from rcmodel import InputScaling
from rcmodel import OptimiseRC


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
    rm_CA = [200, 800]   # [min, max] Capacitance/m2
    C1 = [1.5 * 10 ** 4, 10 ** 6]  # Capacitance
    C2 = [1.5 * 10 ** 4, 10 ** 6]
    R1 = [0.2, 1.2]  # Resistance ((K.m^2)/W)
    R2 = [0.2, 1.2]
    R3 = [0.2, 1.2]
    Rin = [0.2, 1.2]
    cool = [0, 0]  # Cooling limit in W/m2
    gain = [0, 0]  # gain limit (W/m^2) no additional energy

    scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain)

    # Initialise RCModel with the building and InputScaling
    transform = torch.sigmoid
    model = RCModel(building, scaling, dummy_tout, transform)

    model.loads = torch.nn.Parameter(model.loads * 0)  # no heating
    model._build_loads()

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
    rm_CA = [200, 800]  # [min, max] Capacitance/m2
    C1 = [1e3, 1e6]  # Capacitance
    C2 = [1e3, 1e6]
    R1 = [0.1, 1.3]  # Resistance ((K.m^2)/W)
    R2 = [0.1, 1.3]
    R3 = [0.1, 1.3]
    Rin = [0.1, 1.3]
    cool = [0, 10000]  # Cooling limit in W/m2
    gain = [0, 10000]  # gain limit (W/m^2) no additional energy

    scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain)
    transform = torch.sigmoid

    Tin_continuous = None  # not called during test so can get away with None.

    model = RCModel(building_n9, scaling, dummy_tout, transform)

    rm_cap = 500
    ex_cap = [1e4, 2e4]
    ex_r = [0.2, 0.5, 1.2]
    wl_r = 0.8

    original_params = torch.tensor([rm_cap, ex_cap[0], ex_cap[1], ex_r[0], ex_r[1], ex_r[2], wl_r])

    original_loads = 3000 * torch.rand(2, len(model.building.rooms))

    model.params = torch.nn.Parameter(torch.logit(model.scaling.model_param_scaling(original_params)))
    model.loads = torch.nn.Parameter(torch.logit(model.scaling.model_loads_scaling(original_loads)))

    with tempfile.TemporaryDirectory() as tmpdirname:
        num = 0
        # save model values
        model.save(model_id=num, dir_path=tmpdirname)

        del model

        # Initialise new model with different scaling:
        rm_CA = [50, 901]  # [min, max] Capacitance/m2
        C1 = [1e2, 1e5]  # Capacitance
        C2 = [1e2, 1e5]
        R1 = [0.2, 2]  # Resistance ((K.m^2)/W)
        R2 = [0.2, 2]
        R3 = [0.2, 2]
        Rin = [0.2, 2]
        cool = [-5200, 8100]  # Cooling limit in W/m2
        gain = [-5200, 8100]  # gain limit (W/m^2) no additional energy

        scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain)
        transform = torch.sigmoid

        model2 = RCModel(building_n9, scaling, dummy_tout, Tin_continuous, transform)

        path = f"{tmpdirname}/rcmodel{num}.pt"
        model2.load(path=path)

        loaded_params = model2.scaling.physical_param_scaling(transform(model2.params))
        loaded_loads = model2.scaling.physical_loads_scaling(transform(model2.loads))

        diff_params = abs(loaded_params-original_params)
        diff_loads = abs(loaded_loads-original_loads)

    assert (diff_params < 1e-3).all() and (diff_loads < 1e-3).all(), 'model parameters are changing during a save and load'


if __name__ == '__main__':
    pytest.main()

    print("__main__ reached")
