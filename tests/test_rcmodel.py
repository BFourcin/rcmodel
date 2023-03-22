import pytest
import torch
import tempfile
import numpy as np

from rcmodel import RCModel

@pytest.mark.parametrize(
    'model',
    [
        pytest.lazy_fixture('model_n2'),
        pytest.lazy_fixture('model_n9'),
    ], )
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

    output = model(t_eval)

    start_temp = output[0][0, 0].item()  # initial value condition

    assert (not torch.any(output < -5).item()) and (not torch.any(output > start_temp).item()),\
        "Model has gained or lost energy from the system"


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
        num = 0
        # save model values
        path = tmpdirname + '/test_save.pkl'
        model.save(path)

        del model

        # Initialise new model and randomise parameters
        # model2 = model_n9
        # model2.init_physical()

        model2 = RCModel.load(path)

        loaded_params = model2.scaling.physical_param_scaling(model2.transform(model2.params))
        loaded_loads = model2.scaling.physical_loads_scaling(model2.transform(model2.loads))

        diff_params = abs(loaded_params-original_params)
        diff_loads = abs(loaded_loads-original_loads)

    assert (diff_params < 1e-3).all() and (diff_loads < 1e-3).all(), 'model parameters are changing during a save and load'


if __name__ == '__main__':
    pytest.main()
