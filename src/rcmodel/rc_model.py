import torch
import numpy as np
import pandas as pd
import dill
from torch import nn
from torchdiffeq import odeint
from datetime import datetime
from xitorch.interpolate import Interp1D


# TODO: Format comments to be consistent with PEP8.
class RCModel(nn.Module):
    """
    Custom Pytorch model for gradient optimization.
    Initialises with random parameters.

    scaling - Class containing methods to scale inputs from 0-1 back to their usual values and back again.
    transform - function to transform parameters e.g. sigmoid

    Runs using forward method should be sequential as the output is used as the initial condition for the next run, unless manually reset using self.iv
    """

    def __init__(
            self, building, scaling, Tout_continuous, transform=None,
            cooling_policy=None
    ):

        super().__init__()
        self.building = building

        self.transform = transform  # transform performed on parameters e.g. sigmoid
        self.scaling = scaling  # InputScaling class (helper class to go between machine (0-1) and physical values)

        self.Tout_continuous = Tout_continuous  # Interp1D object

        self.cooling_policy = cooling_policy  # Neural net: pi(state) --> action
        self.action = 0  # initialise cooling action

        self.params = None  # initialised in init_physical()
        self.loads = None  # initialised in init_physical()
        self.initialise_parameters()  # initialise params and loads with random numbers

        self.ode_t = None  # Keeps track of t during integration. None is just to initialise attribute
        self.record_action = None  # records Q and t during integration
        self.cool_load = None  # cooling in Watts/m2
        self.gain_load = None  # gain in Watts/m2
        self.t0 = None  # unix epoch start time in seconds
        self.A = None  # System Matrix
        self.B = None  # Input Matrix
        self.iv = None  # initial value
        self.iv_array = None  # Interp1D object of pre found initial values. Means we can get correct iv with just time.

    def setup(self):
        """
        Setup must be called:
            - before the first forward pass.
            - after changing physical parameters (params, loads).
        """

        # These might not have changed from each forward pass but they're cheap enough.
        self._build_matrices()  # get A and B
        self._build_loads()  # get cool and gain load

    def forward(self, t_eval, action=0):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class
        Tout_continuous - A scipy interp1d function covering the whole of t_eval. Tout_continuous(t) = Outside temperature at time t
        iv - Initial value. Starting temperatures of all nodes
        t_eval - times function should return a solution. e.g. torch.arange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """
        self.action = action
        self.record_action = []  # Keeps track of action at time during ODE integration.

        # check if t_eval is formatted correctly:
        if t_eval.dim() != 1:
            t_eval = t_eval.flatten()

        # t0 stores the starting epoch time and t_eval is array of seconds from start, [0, 1*dt, 2*dt, ...]
        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        # THIS IS WRONG SINCE WE ARE NOW CALLING FORWARD MULTIPLE TIMES. IE for each time step.
        # Find the true iv from an initialised Inter1D object.
        # if self.iv_array:
        #     self.iv = self.iv_array(self.t0)

        # Format iv.
        self.iv = self.iv.reshape((2 + len(self.building.rooms), 1)).to(torch.float32)

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(
            self.f_ode, self.iv, t_eval, method="rk4"
        )  # https://github.com/rtqichen/torchdiffeq

        self.iv = None  # Causes error if iv is not reset before next forward pass.

        return integrate  # first two columns are external envelope nodes. i.e not rooms

    def f_ode(self, t, x):
        """
        Provides the function:
        dy/dx = Ax + Bu
        """
        # # get cooling action if policy is not None and 15 minutes has passed since last action
        # if self.cooling_policy:  # policy exists
        #     if t - self.ode_t >= 60*15:
        #         self.action, log_prob = self.cooling_policy.get_action(x[2:], t + self.t0)
        #         self.ode_t = t
        #
        #         if self.cooling_policy.training:  # if in training mode store log_prob
        #             self.cooling_policy.log_probs.append(log_prob)

        if self.cooling_policy:  # policy exists
            # record every time-step
            self.record_action.append(
                [t, self.action]
            )  # This is just used for plotting the cooling after.

        # Get energy input at timestep:
        Q_area = -self.cool_load * self.action  # W/m2
        Q_area = Q_area + self.gain_load  # add the constant gain term
        Q_watts = self.building.proportional_heating(Q_area)

        Tout = self.Tout_continuous(t.item() + self.t0)

        u = self.building.input_vector(Tout, Q_watts)

        return self.A @ x + self.B @ u

    def _build_matrices(self):
        """
        Build/re-build the A and B matrices with the current set of parameters.
        Keep track of parameters used so we can check when to update.
        """
        # Transform parameters
        if self.transform:
            theta = self.transform(self.params)
        else:
            theta = self.params

        # Scale inputs up to their physical values
        theta = self.scaling.physical_param_scaling(theta)

        # Produce matrix A and B from current parameters
        self.A = self.building.update_inputs(theta)
        self.B = self.building.input_matrix()

    def _build_loads(self):
        """
        Transform and scale loads.
        Keep track of loads used so we can only update when there is a difference.
        """
        if self.transform:
            loads = self.transform(self.loads)  # Watts/m2 for cooling and gain.
        else:
            loads = self.loads

        loads = self.scaling.physical_loads_scaling(loads)
        self.cool_load = loads[0, :]
        self.gain_load = loads[1, :]

    def reset_iv(self):
        # Starting Temperatures of nodes. Column vector of shape ([n,1]) n=rooms+2
        self.iv = 26 * torch.ones(2 + len(self.building.rooms))  # set iv at 26 degrees
        # Set iv as column vector. Errors caused if Row vector which are difficult to trace.
        self.iv = self.iv.unsqueeze(1)

    def initialise_parameters(self):
        params = torch.rand(
            self.building.n_params, dtype=torch.float32, requires_grad=True
        )
        loads = torch.rand(
            (2, len(self.building.rooms)), dtype=torch.float32, requires_grad=True
        )

        # enables spread of initial parameters. Otherwise sigmoid(rand) tends towards 0.5.
        if self.transform == torch.sigmoid:
            params = torch.logit(params)  # inverse sigmoid
            loads = torch.logit(loads)

        # make theta torch parameters
        self.params = nn.Parameter(params)

        # initialise the room cooling and gain loads
        self.loads = nn.Parameter(loads)

    def save(self, filename):
        """
        Parameters
        ----------
        filename : str
            Path to save the pickled RCModel to.
        Returns
        ----------
        filename : str
            Keeps format the same as RLLib
        """
        with open(filename, "wb") as dill_file:
            dill.dump(self, dill_file)

        return filename

    @staticmethod
    def load(filename):
        """
        Parameters
        ----------
        filename : str
            Path to the pickled RCModel to load.
        Returns
        ----------
        RCModel
        """
        with open(filename, "rb") as dill_file:
            return dill.load(dill_file)
