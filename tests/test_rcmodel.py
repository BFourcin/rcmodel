import pytest
import torch
import numpy as np

from rcmodel.room import Room
from rcmodel.building import Building
from rcmodel.RCModel import RCModel
from rcmodel.tools import InputScaling


def test_run_model(rooms_n9):

    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    #Initialise Building
    bld = Building(rooms_n9, height, Re, Ce, Rint)

    #Initialise scaling methods:
    rm_CA = [200, 800] #[min, max] Capacitance/area
    ex_C = [1.5*10**4, 10**6] #Capacitance
    R = [0.2, 1.2] # Resistance ((K.m^2)/W)

    scaling = InputScaling(rm_CA, ex_C, R)

    #Function for constant outside temperature. try/except allows for broadcasting of value.
    def dummy_tout(t):
            try:
                return -5*torch.ones(len(t))

            except TypeError:
                return torch.tensor(-5)

    #Initialise RCModel with the building and InputScaling
    transform = torch.sigmoid
    model = RCModel(bld, scaling, dummy_tout, transform)
    model.heating = torch.nn.Parameter(model.heating * 0) #no heating

    #Set parameters for forward run:
    t_eval = torch.arange(0, 200000, 30, dtype=torch.float32)

    output = model(t_eval)


    # from matplotlib import pyplot as plt
    # output = output.squeeze(0)
    # plt.plot(t_eval.detach().numpy(), output[:,0].detach().numpy())
    # plt.show()

    start_temp = output[0][0,0].item() #initial value condition
    print(start_temp)

    assert (not torch.any(output<-5).item()) and (not torch.any(output>start_temp).item()), "Model has gained or lost energy from the system"



if __name__ == '__main__':

    pytest.main()

    print("__main__ reached")
