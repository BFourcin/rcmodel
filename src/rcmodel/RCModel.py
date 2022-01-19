import torch
import numpy as np
from torch import nn
from torchdiffeq import odeint
from datetime import datetime


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

        self.Tout_continuous = Tout_continuous  # Interp1D object

        self.cooling_policy = cooling_policy  # Neural net: pi(state) --> action

        self.params = None  # initialised in init_params()
        self.cooling = None
        self.init_params()  # initialize weights with random numbers, length of params given from building class

        self.ode_t = None  # Keeps track of t during integration. None is just to initialise attribute
        self.record_action = None  # records Q and t during integration
        self.heat = None  # Heat in Watts

        self.reset_iv()  # sets iv

    def forward(self, t_eval):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class
        Tout_continuous - A scipy interp1d function covering the whole of t_eval. Tout_continuous(t) = Outside temperature at time t
        iv - Initial value. Starting temperatures of all nodes
        t_eval - times function should return a solution. e.g. torch.arrange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """
        self.ode_t = -30  # Keeps track of t during ODE integration. Needed to assign log probs only when a step occurs
        self.record_action = []  # Keeps track of action at time during ODE integration.

        # check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        # Transform parameters
        if self.transform:
            theta = self.transform(self.params)
            self.heat = self.transform(self.cooling)  # Watts for each room
        else:
            theta = self.params
            self.heat = self.cooling

        # Scale inputs up to their physical values
        theta = self.scaling.physical_param_scaling(theta)
        self.heat = self.scaling.physical_cooling_scaling(self.heat)

        # t_eval_plus = torch.cat((t_eval, (t_eval[-1] + 600).unsqueeze(0)),
        #                         0)  # added on extra time to avoid error with interp1d(t_end)
        # Qrms_continuous = self.building.Q_continuous(t_eval_plus, Q_avg, self.heating[:, 1], self.heating[:, 2])

        # t0 stores the starting epoch time and t_eval is array of seconds from start, [0, 1*dt, 2*dt, ...]
        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        # Produce matrix A and B from current parameters
        self.A = self.building.update_inputs(theta)
        self.B = self.building.input_matrix()

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(self.fODE, self.iv, t_eval, method='rk4')  # https://github.com/rtqichen/torchdiffeq

        return integrate  # first two columns are external envelope nodes. i.e not rooms

    def fODE(self, t, x):
        """
        Provides the function:
        dy/dx = Ax + Bu
        """

        mu = 23.359
        std_dev = 1.41

        x_norm = (x - mu) / std_dev
        x_norm = torch.reshape(x_norm, (-1,))

        day = 24 * 60 ** 2
        week = 7 * day
        # year = (365.2425) * day

        time_signals = [np.sin((t+self.t0) * (2 * np.pi / day)),
                        np.cos((t+self.t0) * (2 * np.pi / day)),
                        np.sin((t+self.t0) * (2 * np.pi / week)),
                        np.cos((t+self.t0) * (2 * np.pi / week))
                        ]

        state_cool = torch.cat((x_norm, torch.tensor(time_signals, dtype=torch.float32)))

        # get cooling action if policy is not None and 15 minutes has passed since last action
        if self.cooling_policy and t - self.ode_t >= 60*15:
            action, log_prob = self.cooling_policy.get_action(state_cool)
            self.ode_t = t
            self.record_action.append([t, action])  # This is just used for plotting the cooling after.

            if self.cooling_policy.training:  # if in training mode store log_prob
                self.cooling_policy.log_probs.append(log_prob)

        else:  # No cooling
            action = 0

        Q = -self.heat * action

        # Q + gain
        Q = Q + self.building.proportional_heating(self.building.gain)

        Tout = self.Tout_continuous(t.item() + self.t0)

        u = self.building.input_vector(Tout, Q)

        return self.A @ x + self.B @ u

    def reset_iv(self):
        # Starting Temperatures of nodes. Column vector of shape ([n,1]) n=rooms+2
        self.iv = 26 * torch.ones(2 + len(self.building.rooms))  # set iv at 26 degrees
        # Set iv as column vector. Errors caused if Row vector which are difficult to trace.
        self.iv = self.iv.unsqueeze(1)

    def init_params(self):
        params = torch.rand(self.building.n_params, dtype=torch.float32, requires_grad=True)

        # enables spread of initial parameters. Otherwise sigmoid(rand) tends towards 0.5.
        if self.transform == torch.sigmoid:
            params = torch.logit(params)  # inverse sigmoid

        # make theta torch parameters
        self.params = nn.Parameter(params)
        self.cooling = nn.Parameter(torch.rand((len(self.building.rooms)), dtype=torch.float32, requires_grad=True))  # [Q]

    def save(self, model_id=0, dir_path=None):
        """
        Save the model, but first transform parameters back to their physical values.
        This makes passing between models with different scaling limits possible.
        Use load method to load back in. Parameters will be converted back to 0-1 using the current scaling function.
        """
        scaled_state_dict = self.state_dict()
        scaled_state_dict['params'] = self.scaling.physical_param_scaling(self.transform(scaled_state_dict['params']))
        scaled_state_dict['cooling'] = self.scaling.physical_cooling_scaling(self.transform(scaled_state_dict['cooling']))

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

        scaled_state_dict['cooling'] = torch.logit(
            self.scaling.model_cooling_scaling(scaled_state_dict['cooling']) + 1e-15
        )

        self.load_state_dict(scaled_state_dict)
