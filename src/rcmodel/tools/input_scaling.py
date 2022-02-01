from ..physical import Building
import torch


class InputScaling(Building):
    """
    Class to group methods for scaling input parameters.
    Provide the limits for each variable. e.g. rm_CA = [100, 1000]

    Model scaling - scales physical parameters to between 0-1 using min max.
    Physical scaling - returns parameters back to their physical meaning.

    Initialise with:
    InputScaling(rm_CA ex_C, R, Q_limit)
    or
    InputScaling(input_range=[rm_CA, ex_C, R, Q_limit])
    """

    def __init__(self, rm_CA=None, ex_C=None, R=None, Q_limit=None, input_range=None):

        if input_range is None:
            input_range = [rm_CA, ex_C, R, Q_limit]  # Q_limit is in W/m2

            # check for None
            if None in input_range:
                raise ValueError('None type in list. Define all inputs or use a predefined input_range instead')

        self.input_range = torch.tensor(input_range)

    def physical_param_scaling(self, theta_scaled):
        """
        Scale from 0-1 back to physical value.
        """

        rm_cap, ex_cap, ex_r, wl_r, gain = self.categorise_theta(theta_scaled)

        rm_cap = self.unminmaxscale(rm_cap, self.input_range[0])
        ex_cap = self.unminmaxscale(ex_cap, self.input_range[1])
        ex_r   = self.unminmaxscale(ex_r,   self.input_range[2])
        wl_r   = self.unminmaxscale(wl_r,   self.input_range[2])
        gain   = self.unminmaxscale(gain,   self.input_range[3])

        if wl_r.ndim == 0:
            wl_r = torch.unsqueeze(wl_r, 0)
        if gain.ndim == 0:
            gain = torch.unsqueeze(gain, 0)

        theta = torch.cat([rm_cap, ex_cap, ex_r, wl_r, gain])

        return theta

    def model_param_scaling(self, theta):
        """
        Scale to 0-1.
        """

        rm_cap, ex_cap, ex_r, wl_r, gain = self.categorise_theta(theta)

        rm_cap = self.minmaxscale(rm_cap, self.input_range[0])
        ex_cap = self.minmaxscale(ex_cap, self.input_range[1])
        ex_r   = self.minmaxscale(ex_r  , self.input_range[2])
        wl_r   = self.minmaxscale(wl_r  , self.input_range[2])
        gain   = self.minmaxscale(gain  , self.input_range[3])

        if wl_r.ndim == 0:
            wl_r = torch.unsqueeze(wl_r, 0)
        if gain.ndim == 0:
            gain = torch.unsqueeze(gain, 0)

        theta_scaled = torch.cat([rm_cap, ex_cap, ex_r, wl_r, gain])

        return theta_scaled

    def physical_cooling_scaling(self, Q):
        """
        Scale from 0-1 back to physical value in W/m2
        """

        Q_area = self.unminmaxscale(Q, [0, self.input_range[3].max()])

        return Q_area

    def model_cooling_scaling(self, Q_area):
        """
        Scale to 0-1.
        """

        Q = self.minmaxscale(Q_area, [0, self.input_range[3].max()])

        return Q

    def minmaxscale(self, x, x_range):
        if not torch.is_tensor(x_range):
            x_range = torch.tensor(x_range)
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x_scaled = (x - x_range.min()) / (x_range.max() - x_range.min())

        return x_scaled

    def unminmaxscale(self, x_scaled, x_range):
        if not torch.is_tensor(x_range):
            x_range = torch.tensor(x_range)
        if not torch.is_tensor(x_scaled):
            x_scaled = torch.tensor(x_scaled)

        x = x_scaled * (x_range.max() - x_range.min()) + x_range.min()

        return x

