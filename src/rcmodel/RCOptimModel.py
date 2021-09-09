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
        self.building = building

        self.transform = transform # transform - on parameters e.g. sigmoid

        self.scaling = scaling #InputScaling class

        # initialize weights with random numbers, length of params given from building class
        params = torch.rand(building.n_params, dtype=torch.float32, requires_grad=True)
        # make theta torch parameters
        self.params = nn.Parameter(params)
        self.heating = nn.Parameter(torch.randn((len(building.rooms), 3), dtype=torch.float32, requires_grad=True)) # [Q, theta_A, theta_B]

        self.Q_lim = 500

    def forward(self, Tout_continuous, iv, t_eval):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class
        Tout_continuous - A scipy interp1d function covering the whole of t_eval. Tout_continuous(t) = Outside temperature at time t
        iv - Inital value. Starting temperatures of all nodes
        t_eval - times function should return a solution. e.g. torch.arange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """

        if self.transform:
            theta = self.transform(self.params)
            Q_avg = self.transform(self.heating[:,0]) #W/m2 for each room
        else:
            theta = self.params
            Q_avg = self.heating[:,0]

        # Scale inputs up to their physical values
        theta   = self.scaling.physical_scaling(theta)
        Q_avg   = self.scaling.heat_scaling(Q_avg, Q_lim=self.Q_lim)
        
        # Get iterp1d object of Q
        # theta_A = self.heating[:,1]
        # theta_B = self.heating[:,2]


        t_eval_plus = torch.cat((t_eval, (t_eval[-1]+600).unsqueeze(0)), 0)
        Qrms_continuous = self.building.Q_continuous(t_eval_plus, Q_avg, self.heating[:,1], self.heating[:,2])

        t0 = t_eval[0]
        t_eval = t_eval - t0


        # ensures iv has correct dimensions. If iv is a row vector try statement catches and converts to column vector.
        # If iv has incorrect dim it causes error downstream which is difficult to trace back.
        try:
            if iv.shape[0] != 2+len(self.building.rooms) or iv.shape[1] != 1:
                raise ValueError('iv has incorrect dimensions, should be torch.Size([n, 1])')
        except IndexError:
            iv = iv.unsqueeze(1)

        xdot = self.ODE(theta, self.building, Tout_continuous, Qrms_continuous, t0)
        fODE = xdot.fODE

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(fODE, iv, t_eval, method='rk4') #https://github.com/rtqichen/torchdiffeq

        return integrate


    class ODE():
        """
        Class to provide the function:
        dy/dx = Ax + Bu
        """

        def __init__(self, theta, building, Tout_continuous, Qrms_continuous, t0):

            self.building = building
            self.A = self.building.update_inputs(theta) #update building and get matrix A
            self.B = building.input_matrix()
            self.Tout_continuous = Tout_continuous
            self.Qrms_continuous = Qrms_continuous
            self.t0 = t0 #starting time


        def fODE(self, t, x):
            Tout = self.Tout_continuous(t.item() + self.t0)

            Q = self.Qrms_continuous(t + self.t0)
            # print(t)

            u = self.building.input_vector(Tout, Q)
            # print('u' , u)
            # print('A@x ', self.A @ x)
            # print('B@u', self.B @ u)

            return self.A @ x + self.B @ u
