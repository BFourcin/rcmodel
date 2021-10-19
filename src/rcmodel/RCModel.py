import torch
from torch import nn
from torchdiffeq import odeint
from datetime import datetime


class RCModel(nn.Module):
    """
    Custom Pytorch model for gradient optimization.
    Initilaises with random parameters.

    scaling - Class containing methods to scale inputs from 0-1 back to their usual values and back again.
    transform - function to transform parameters e.g. sigmoid

    Runs using foward method should be sequential as the output is used as the inital conditon for the next run, unless manually reset using self.iv
    """
    def __init__(self, building, scaling, Tout_continuous, transform=None, cooling_policy=None):

        super().__init__()
        self.building = building

        self.transform = transform # transform - on parameters e.g. sigmoid

        self.scaling = scaling #InputScaling class (helper class to go between machine (0-1) and physical values)

        self.Tout_continuous = Tout_continuous #Interp1D object

        self.cooling_policy = cooling_policy #Neural net: cooling_policy(state) --> action

        # initialize weights with random numbers, length of params given from building class
        params = torch.rand(building.n_params, dtype=torch.float32, requires_grad=True)
        # make theta torch parameters
        self.params = nn.Parameter(params)
        self.heating = nn.Parameter(torch.randn((len(building.rooms), 3), dtype=torch.float32, requires_grad=True)) # [Q, theta_A, theta_B]

        self.Q_lim = 500

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.reset_iv() #sets iv

        

    def forward(self, t_eval):
        """
        Integrates the ode forward in time.

        building - Initialised RCModel Class
        Tout_continuous - A scipy interp1d function covering the whole of t_eval. Tout_continuous(t) = Outside temperature at time t
        iv - Inital value. Starting temperatures of all nodes
        t_eval - times function should return a solution. e.g. torch.arange(0, 10000, 30). ensure dtype=float32
        t0 - starting time if not 0
        """

        #check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        # Check if Cuda is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Transform parameters
        if self.transform:
            theta = self.transform(self.params)
            Q_avg = self.transform(self.heating[:,0]) #W/m2 for each room
        else:
            theta = self.params
            Q_avg = self.heating[:,0]

        # Scale inputs up to their physical values
        theta   = self.scaling.physical_scaling(theta)
        Q_avg   = self.scaling.heat_scaling(Q_avg, Q_lim=self.Q_lim)

        t_eval_plus = torch.cat((t_eval, (t_eval[-1]+600).unsqueeze(0)), 0) #added on extra time to avoide error with interp1d(t_end)
        Qrms_continuous = self.building.Q_continuous(t_eval_plus, Q_avg, self.heating[:,1], self.heating[:,2])

        # t0 stores the starting epoch time and t_eval is array of seconds from start, [0, 1*dt, 2*dt, ...]
        self.t0 = t_eval[0]
        t_eval = t_eval - self.t0

        #Produce matrix A and B from current parameters
        self.A = self.building.update_inputs(theta).to(self.device)
        self.B = self.building.input_matrix().to(self.device)

        # integrate using fixed step (rk4) see torchdiffeq docs for more options.
        integrate = odeint(self.fODE, self.iv, t_eval, method='rk4') #https://github.com/rtqichen/torchdiffeq

        #get last ouput and use for next initial value
        self.iv = integrate[-1, :]

        return integrate[:,2:] #first two columns are removed, they're external wall nodes.



    def decimal_time(self, unix_epoch):
        """Time to percentage of year, week and day"""

        date = datetime.fromtimestamp(unix_epoch)
        _, isoweek, isoweekday = date.isocalendar()

        time_of_year = isoweek/52
        time_of_week = (isoweekday-1)/7 + (date.hour*60**2 + date.minute*60 + date.second)/(60**2*24*7)
        time_of_day = (date.hour*60**2 + date.minute*60 + date.second)/(60**2*24)

        return time_of_year, time_of_week, time_of_day



    def fODE(self, t, x):
        """
        Provides the function:
        dy/dx = Ax + Bu
        """

        # cool_param: [x , time_of_week, time_of_day]


        time_of_year, time_of_week, time_of_day = self.decimal_time(t.item() + self.t0.item())

        mu = 23.359
        std_dev = 1.41

        x_norm = (x[2:]-mu)/std_dev
        x_norm = torch.reshape(x_norm, (-1,))

        cool_param = torch.cat((x_norm, torch.tensor([time_of_week, time_of_day], device=self.device)))

        value = -16914
        if self.cooling_policy:
            Q = value * self.cooling_policy.predict(cool_param.float())
        else:
            Q = torch.zeros(len(self.building.rooms))

        # Q = self.Qrms_continuous(t + self.t0)

        Tout = self.Tout_continuous(t.item() + self.t0)
        u = self.building.input_vector(Tout, Q)

        return self.A @ x + self.B @ u


    def reset_iv(self):
        #Starting Temperatures of nodes. Column vector of shape ([n,1]) n=rooms+2
        self.iv = 26 * torch.ones(2+len(self.building.rooms), device=self.device) #set iv at 26 degrees
        self.iv = self.iv.unsqueeze(1) #set iv as column vector. Errors caused if Row vector which are difficult to trace.
