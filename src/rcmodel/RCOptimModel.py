import torch
from torch import nn
from torchdiffeq import odeint

class RCOptimModel(nn.Module):
    """
    Custom Pytorch model for gradient optimization.
    Initilaises with random parameters.

    scaling - Class containing methods to scale inputs from 0-1 back to their usual values and back again.
    transform - function to transform parameters e.g. sigmoid
    """
    def __init__(self, building, scaling, transform=None):

        super().__init__()
        # initialize weights with random numbers, length of params given from building class
        params = torch.rand(building.n_params, dtype=torch.float32, requires_grad=True)
        # make theta torch parameters
        self.params = nn.Parameter(params)

        self.building = building

        self.transform = transform # transform - on parameters e.g. sigmoid

        self.scaling = scaling #InputScaling class


    def forward(self, Tout_continous, iv, t_eval, t0=0):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class
        Tout_continous - A scipy interp1d function covering the whole of t_eval. Tout_continous(t) = Outside temperature at time t
        iv - Inital value. Starting temperatures of all nodes
        t_eval - times function should return a solution. e.g. torch.arange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """

        if self.transform:
            x = self.transform(self.params)
        else:
            x = self.params

        theta = self.scaling.physical_scaling(x)

        # ensures iv has correct dimensions. If iv is a row vector try statement catches and converts to column vector.
        # If iv has incorrect dim then causes error downstream which is difficult to trace back.
        try:
            if iv.shape[0] != 2+len(self.building.rooms) or iv.shape[1] != 1:
                raise ValueError('iv has incorrect dimensions, should be torch.Size([n, 1])')
        except IndexError:
            iv = iv.unsqueeze(1)

        xdot = self.ODE(theta, self.building, Tout_continous, t0)
        fODE = xdot.fODE

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(fODE, iv, t_eval, method='rk4') #https://github.com/rtqichen/torchdiffeq

        return integrate


    class ODE():
        """
        Class to provide the function:
        dy/dx = Ax + Bu
        """

        def __init__(self, theta, building, Tout_continous, t0):

            self.building = building
            self.A, self.Q = self.building.update_inputs(theta) #update building and get matrix A and Q for current theta
            self.B = building.input_matrix()
            self.Tout_continous = Tout_continous
            self.t0 = t0 #starting time

        def fODE(self, t, x):

            u = self.building.input_vector(self.Tout_continous(t.item() + self.t0), self.Q)

            return self.A @ x + self.B @ u
