import torch
import numpy as np
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
        self.Tout_continuous = Tout_continuous  # Interp1D object
        self.cooling_policy = cooling_policy  # Neural net: pi(state) --> action
        self.action = 0  # initialise cooling action

        self.params = None  # initialised in init_physical()
        self.loads = None  # initialised in init_physical()
        self.init_physical()  # initialise with random numbers, length of params given from building class

        self.ode_t = None  # Keeps track of t during integration. None is just to initialise attribute
        self.record_action = None  # records Q and t during integration
        self.cool_load = None  # cooling in Watts/m2
        self.gain_load = None  # gain in Watts/m2
        self.t0 = None  # unix epoch start time in seconds
        self.A = None  # System Matrix
        self.B = None  # Input Matrix
        self.iv = None  # initial value
        self.iv_array = None  # Interp1D object of pre found initial values. Means we can get correct iv with just time.
        self.Tin_continuous = None

        self.reset_iv()  # sets iv

    def forward(self, t_eval):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class
        Tout_continuous - A scipy interp1d function covering the whole of t_eval. Tout_continuous(t) = Outside temperature at time t
        iv - Initial value. Starting temperatures of all nodes
        t_eval - times function should return a solution. e.g. torch.arange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """
        self.ode_t = -900  # keeps track of previous time an action was produced. Initialised to work at t=0.
        self.record_action = []  # Keeps track of action at time during ODE integration.

        # check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        # Transform parameters
        if self.transform:
            theta = self.transform(self.params)
            loads = self.transform(self.loads)  # Watts/m2 for cooling and gain.
        else:
            theta = self.params
            loads = self.loads

        # Scale inputs up to their physical values
        theta = self.scaling.physical_param_scaling(theta)
        loads = self.scaling.physical_cooling_scaling(loads)

        self.cool_load = loads[0, :]
        self.gain_load = loads[1, :]

        # t_eval_plus = torch.cat((t_eval, (t_eval[-1] + 600).unsqueeze(0)),
        #                         0)  # added on extra time to avoid error with interp1d(t_end)
        # Qrms_continuous = self.building.Q_continuous(t_eval_plus, Q_avg, self.heating[:, 1], self.heating[:, 2])

        # t0 stores the starting epoch time and t_eval is array of seconds from start, [0, 1*dt, 2*dt, ...]
        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        # Produce matrix A and B from current parameters
        self.A = self.building.update_inputs(theta)
        self.B = self.building.input_matrix()

        # Find the true iv from an initialised Inter1D object.
        if self.iv_array:
            self.iv = self.iv_array(self.t0).unsqueeze(0).T

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(self.f_ode, self.iv, t_eval, method='rk4')  # https://github.com/rtqichen/torchdiffeq

        return integrate  # first two columns are external envelope nodes. i.e not rooms

    def f_ode(self, t, x):
        """
        Provides the function:
        dy/dx = Ax + Bu
        """

        # get cooling action if policy is not None and 15 minutes has passed since last action
        if self.cooling_policy:  # policy exists
            if t - self.ode_t >= 60*15:
                self.action, log_prob = self.cooling_policy.get_action(x[2:], t + self.t0)
                self.ode_t = t

                if self.cooling_policy.training:  # if in training mode store log_prob
                    self.cooling_policy.log_probs.append(log_prob)

            # record every time-step
            self.record_action.append([t, self.action])  # This is just used for plotting the cooling after.

        # Get energy input at timestep:
        Q_area = -self.cool_load * self.action  # W/m2
        Q_area = Q_area + self.gain_load  # add the constant gain term
        Q_watts = self.building.proportional_heating(Q_area)

        Tout = self.Tout_continuous(t.item() + self.t0)

        u = self.building.input_vector(Tout, Q_watts)

        return self.A @ x + self.B @ u

    def get_iv_array(self, t_eval):
        """
        Perform the integration but force outside and inside temperature to be from data.
        Only the latent temperature nodes in the external walls are free to change meaning we can find out their true
        values for a given model.
        """
        with torch.no_grad():
            # Transform parameters
            if self.transform:
                theta = self.transform(self.params)
            else:
                theta = self.params

            # Update building with current parameters
            theta = self.scaling.physical_param_scaling(theta)
            _ = self.building.update_inputs(theta)

            bl = self.building

            # Recalculate the A & B matrices. We could chop and re jig from the full matrices, but it is not super
            # simple so recalculating is less risky.
            A = torch.zeros([2, 2])
            A[0, 0] = bl.surf_area * (-1 / (bl.Re[0] * bl.Ce[0]) - 1 / (bl.Re[1] * bl.Ce[0]))
            A[0, 1] = bl.surf_area / (bl.Re[1] * bl.Ce[0])
            A[1, 0] = bl.surf_area / (bl.Re[1] * bl.Ce[1])
            A[1, 1] = bl.surf_area * (-1 / (bl.Re[1] * bl.Ce[1]) - 1 / (bl.Re[2] * bl.Ce[1]))

            B = torch.zeros([2, 2])
            B[0, 0] = bl.surf_area / (bl.Re[0] * bl.Ce[0])
            B[1, 1] = bl.surf_area / (bl.Re[2] * bl.Ce[1])

            # check if t_eval is formatted correctly:
            if t_eval.dim() > 1:
                t_eval = t_eval.squeeze(0)

            avg_tout = self.Tout_continuous(t_eval).mean()
            avg_tin = self.Tin_continuous(t_eval).mean()

            t0 = t_eval[0]
            t_eval = t_eval - t0

            self.iv = self.steady_state_iv(avg_tout, avg_tin)  # Use avg temp as a good guess for iv.

            def latent_f_ode(t, x):
                Tout = self.Tout_continuous(t.item() + t0)
                Tin = self.Tin_continuous(t.item() + t0)

                u = torch.tensor([[Tout],
                                  [Tin]])

                return A @ x + B @ u.to(torch.float32)

            integrate = odeint(latent_f_ode, self.iv[0:2], t_eval,
                               method='rk4')  # https://github.com/rtqichen/torchdiffeq

            integrate = integrate.squeeze()

            # Add on inside temperature data to be used to initialise rooms at the correct temp.
            iv_array = torch.empty(len(integrate), len(bl.rooms) + 2)
            iv_array[:, 0:2] = integrate
            iv_array[:, 2:] = self.Tin_continuous(t_eval + t0).T
            iv_array = Interp1D(t_eval+t0, iv_array.T, method='linear')

        return iv_array

    def reset_iv(self):
        # Starting Temperatures of nodes. Column vector of shape ([n,1]) n=rooms+2
        self.iv = 26 * torch.ones(2 + len(self.building.rooms))  # set iv at 26 degrees
        # Set iv as column vector. Errors caused if Row vector which are difficult to trace.
        self.iv = self.iv.unsqueeze(1)

    def steady_state_iv(self, temp_out, temp_in):
        """
        Calculate the initial conditions of the latent variables given a steady state indoor and outdoor temperature.
        Initial values of room nodes are set to temp in.

        temp_out: float
            Steady state outside temperature.
        temp_in: tensor
            Steady state inside temperature. len(temp_in) = n_rooms
        :return: tensor
            Column tensor of initial values at each node.
        """
        I = (temp_out - temp_in) / sum(self.building.Re)  # I=V/R
        v1 = temp_out - I * self.building.Re[0]
        v2 = v1 - I * self.building.Re[1]

        iv = torch.tensor([[v1], [v2], [temp_in]]).to(torch.float32)

        return iv

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

    def save(self, model_id=0, dir_path=None):
        """
        Save the model, but first transform parameters back to their physical values.
        This makes passing between models with different scaling limits possible.
        Use load method to load back in. Parameters will be converted back to 0-1 using the current scaling function.
        """
        scaled_state_dict = self.state_dict()
        scaled_state_dict['params'] = self.scaling.physical_param_scaling(self.transform(scaled_state_dict['params']))
        scaled_state_dict['loads'] = self.scaling.physical_cooling_scaling(self.transform(scaled_state_dict['loads']))

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
            self.scaling.model_cooling_scaling(scaled_state_dict['loads']) + 1e-15
        )

        self.load_state_dict(scaled_state_dict)
