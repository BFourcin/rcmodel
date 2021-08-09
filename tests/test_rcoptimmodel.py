import pytest
import torch
import numpy as np

from rcmodel.room import Room
from rcmodel.RCModel import RCModel
from rcmodel.RCOptimModel import RCOptimModel
from rcmodel.tools import InputScaling


def test_run_model(rooms_n9):

    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    #Initialise RCModel
    bld = RCModel(rooms_n9, height, Re, Ce, Rint)

    #Initialise RCOptimModel with the building
    model = RCOptimModel(bld)

    #Initialise scaling methods:
    rm_CA = [200, 800] #[min, max] Capacitance/area
    ex_C = [1.5*10**4, 10**6] #Capacitance
    R = [0.2, 1.2] # Resistance ((K.m^2)/W)
    #SET HEATING TO ZERO
    Q_avg = [0, 0] # Heat/area (W/m^2)

    scaling = InputScaling(rm_CA, ex_C, R, Q_avg)
    scale_f = scaling.physical_scaling

    #Set hyper parameters for forward run:
    iv = 20*torch.ones((2+len(bld.rooms),1))
    t_eval = torch.arange(0, 200000, 30, dtype=torch.float32)

    #Function for constant outside temperature. try/except allows for broadcasting of value.
    def dummy_tout(t):
            try:
                return -5*torch.ones(len(t))

            except TypeError:
                return -5


    output= model.forward(dummy_tout, iv, t_eval, scale_f, torch.tensor(0))

    assert (not torch.any(output<-5).item()) and (not torch.any(output>20).item()), "Model has gained or lost energy from the system"
