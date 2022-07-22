import torch
import numpy as np
import pandas as pd
from torch import nn
from torchdiffeq import odeint
from datetime import datetime
from xitorch.interpolate import Interp1D


class RCModel(nn.Module):
    """
    Custom Pytorch model for gradient optimization.
    Initialises with random parameters.

    scaling - Class containing methods to scale inputs from 0-1 back to their usual values and back again.
    transform - function to transform parameters e.g. sigmoid

    Runs using forward method should be sequential as the output is used as the initial condition for the next run, unless manually reset using self.iv
    """

    def __init__(self, building, scaling, Tout_continuous, transform=None, cooling_policy=None):

        super().__init__()
        self.building = building

        self.transform = transform  # transform performed on parameters e.g. sigmoid
        self.scaling = scaling  # InputScaling class (helper class to go between machine (0-1) and physical values)

        # df = pd.read_csv(weather_data_path)
        # Tout = torch.tensor(df['Hourly Temperature (°C)'])
        # t = torch.tensor(df['time'])
        self.Tout_continuous = Tout_continuous  # Interp1D object

        # time_data, temp_data = room_dataset.get_all_data()
        # self.Tin_continuous = Interp1D(time_data, temp_data[:, 0:len(self.building.rooms)].T, method='linear')

        self.cooling_policy = cooling_policy  # Neural net: pi(state) --> action
        self.action = 0  # initialise cooling action

        self.params = None  # initialised in init_physical()
        self.loads = None  # initialised in init_physical()
        self.params_old = None  # copies used to check when parameters have been updated
        self.loads_old = None
        self.init_physical()  # initialise params and loads with random numbers

        self.ode_t = None  # Keeps track of t during integration. None is just to initialise attribute
        self.record_action = None  # records Q and t during integration
        self.cool_load = None  # cooling in Watts/m2
        self.gain_load = None  # gain in Watts/m2
        self.t0 = None  # unix epoch start time in seconds
        self.A = None  # System Matrix
        self.B = None  # Input Matrix
        self.iv = None  # initial value
        self.iv_array = None  # Interp1D object of pre found initial values. Means we can get correct iv with just time.

        self._build_loads()  # get cool and gain load
        self.reset_iv()  # sets iv

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

        # self.ode_t = -900  # keeps track of previous time an action was produced. Initialised to work at t=0.
        self.record_action = []  # Keeps track of action at time during ODE integration.
        iv_note = False

        # If physical parameters have changed we need to do two things: update Matrices A & B and recalculate iv_array.
        # It doesn't matter if loads have changed as they do not affect A/B or iv.
        if not torch.eq(self.params_old, self.params).all():
            self._build_matrices()  # Get A and B matrix of current parameters
            iv_note = True

        if not torch.eq(self.loads_old, self.loads).all():
            self._build_loads()

        # check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        # t0 stores the starting epoch time and t_eval is array of seconds from start, [0, 1*dt, 2*dt, ...]
        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        # Find the true iv from an initialised Inter1D object.
        # if self.iv_array:
        #     self.iv = self.iv_array(self.t0).unsqueeze(0).T
        #     if iv_note: print('iv_array not currently valid, check code.')

        if self.iv.dtype != torch.float32:
            self.iv = self.iv.to(torch.float32)

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(self.f_ode, self.iv, t_eval, method='rk4')  # https://github.com/rtqichen/torchdiffeq

        self.iv = None  # Ensures iv is set between forward calls.

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
            self.record_action.append([t, self.action])  # This is just used for plotting the cooling after.

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

        # create copy of current parameters
        self.params_old = self.params.detach().clone()

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

        self.loads_old = self.loads.detach().clone()

    def reset_iv(self):
        # Starting Temperatures of nodes. Column vector of shape ([n,1]) n=rooms+2
        self.iv = 26 * torch.ones(2 + len(self.building.rooms))  # set iv at 26 degrees
        # Set iv as column vector. Errors caused if Row vector which are difficult to trace.
        self.iv = self.iv.unsqueeze(1)

    def init_physical(self):
        params = torch.rand(self.building.n_params, dtype=torch.float32, requires_grad=True)
        loads = torch.rand((2, len(self.building.rooms)), dtype=torch.float32, requires_grad=True)

        # enables spread of initial parameters. Otherwise sigmoid(rand) tends towards 0.5.
        if self.transform == torch.sigmoid:
            params = torch.logit(params)  # inverse sigmoid
            loads = torch.logit(loads)

        # make theta torch parameters
        self.params = nn.Parameter(params)

        # initialise the room cooling and gain loads
        self.loads = nn.Parameter(loads)

        # these are just used to check if parameters have changed between runs. Initialised to dummy value
        self.params_old = torch.tensor(torch.nan)
        self.loads_old = torch.tensor(torch.nan)

    def save(self, model_id=0, dir_path=None):
        """
        Save the model, but first transform parameters back to their physical values.
        This makes passing between models with different scaling limits possible.
        Use load method to load back in. Parameters will be converted back to 0-1 using the current scaling function.
        """
        scaled_state_dict = self.state_dict()
        scaled_state_dict['params'] = self.scaling.physical_param_scaling(self.transform(scaled_state_dict['params']))
        scaled_state_dict['loads'] = self.scaling.physical_loads_scaling(self.transform(scaled_state_dict['loads']))

        if dir_path is None:
            dir_path = "./outputs/models/"

        from pathlib import Path
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        torch.save(scaled_state_dict, dir_path + "/rcmodel" + str(model_id) + ".pt")

    def load(self, path):
        """
        Load in model and convert parameters from their physical values to 0-1 using scaling methods.
        Only use if model has been saved with self.save() method.
        """
        scaled_state_dict = torch.load(path)

        # plus 1e-15 used to stop log(0)=-inf
        scaled_state_dict['params'] = torch.logit(
            self.scaling.model_param_scaling(scaled_state_dict['params']) + 1e-15
        )

        scaled_state_dict['loads'] = torch.logit(
            self.scaling.model_loads_scaling(scaled_state_dict['loads']) + 1e-15
        )

        self.load_state_dict(scaled_state_dict)
