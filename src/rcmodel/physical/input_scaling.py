from . import Building
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

    def __init__(self, C_rm=None, C1=None, C2=None, R1=None, R2=None, R3=None, Rin=None, cool=None, gain=None):

        self.phys_param_range = [C_rm, C1, C2, R1, R2, R3, Rin]
        self.energy_param_range = [cool, gain]

        # check if the ranges are in the correct format. i.e. [lb, ub]
        for ranges in [self.phys_param_range, self.energy_param_range]:
            for bounds in ranges:
                assert bounds[0] <= bounds[1], f'Range for each parameter should be in the form [lb, ub], got: {bounds}'

    def physical_param_scaling(self, theta_scaled):
        """
        Scale from 0-1 back to physical value.
        """

        theta = self.unminmaxscale(theta_scaled, self.phys_param_range)

        # rm_cap, ex_cap, ex_r, wl_r = self.categorise_theta(theta_scaled)
        #
        # rm_cap = self.unminmaxscale(rm_cap, self.input_range[0])
        # ex_cap = self.unminmaxscale(ex_cap, self.input_range[1])
        # ex_r = self.unminmaxscale(ex_r, self.input_range[2])
        # wl_r = self.unminmaxscale(wl_r, self.input_range[2])
        #
        # if wl_r.ndim == 0:
        #     wl_r = torch.unsqueeze(wl_r, 0)
        #
        # theta = torch.cat([rm_cap, ex_cap, ex_r, wl_r])

        return theta

    def model_param_scaling(self, theta):
        """
        Scale to 0-1.
        """

        theta_scaled = self.minmaxscale(theta, self.phys_param_range)

        # rm_cap, ex_cap, ex_r, wl_r = self.categorise_theta(theta)
        #
        # rm_cap = self.minmaxscale(rm_cap, self.input_range[0])
        # ex_cap = self.minmaxscale(ex_cap, self.input_range[1])
        # ex_r = self.minmaxscale(ex_r, self.input_range[2])
        # wl_r = self.minmaxscale(wl_r, self.input_range[2])
        #
        # if wl_r.ndim == 0:
        #     wl_r = torch.unsqueeze(wl_r, 0)
        #
        # theta_scaled = torch.cat([rm_cap, ex_cap, ex_r, wl_r])

        return theta_scaled

    def physical_loads_scaling(self, loads_m):
        """
        Scale from 0-1 back to physical value in W/m2
        """
        loads_r = torch.zeros(loads_m.shape)
        loads_r[0, :] = self.unminmaxscale(loads_m[0, :], self.energy_param_range[0])
        loads_r[1, :] = self.unminmaxscale(loads_m[1, :], self.energy_param_range[1])

        # Q_area = self.unminmaxscale(Q, [0, self.input_range[3].max()])

        return loads_r

    def model_loads_scaling(self, loads_r):
        """
        Scale to 0-1.
        """
        loads_m = torch.zeros(loads_r.shape)
        loads_m[0, :] = self.minmaxscale(loads_r[0, :], self.energy_param_range[0])
        loads_m[1, :] = self.minmaxscale(loads_r[1, :], self.energy_param_range[1])

        # Q = self.minmaxscale(Q_area, [0, self.input_range[3].max()])

        return loads_m

    def minmaxscale(self, x, x_range):
        if not torch.is_tensor(x_range):
            x_range = torch.tensor(x_range)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x_range.ndim == 1:
            x_range = x_range.unsqueeze(0)

        # check if x too small (~ inverses)
        assert ~(x < x_range[:, 0]).any(), f'From model inputs: {x}, input ' \
                                           f'{x[(x < x_range[:, 0])]} is outside defined range: \n{x_range} '
        # check if x too big
        assert ~(x > x_range[:, 1]).any(), f'From model inputs: {x}, input ' \
                                           f'{x[(x > x_range[:, 1])]} is outside defined range: \n{x_range} '

        x_scaled = (x - x_range[:, 0]) / (x_range[:, 1] - x_range[:, 0])

        return x_scaled

    def unminmaxscale(self, x_scaled, x_range):
        if not torch.is_tensor(x_range):
            x_range = torch.tensor(x_range)
        if not torch.is_tensor(x_scaled):
            x_scaled = torch.tensor(x_scaled)
        if x_range.ndim == 1:
            x_range = x_range.unsqueeze(0)

        x = x_scaled * (x_range[:, 1] - x_range[:, 0]) + x_range[:, 0]

        return x
