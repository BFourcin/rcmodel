import torch

from . import Building


class InputScaling(Building):
    """
    Class to group methods for scaling input parameters.
    Provide the limits for each variable. e.g. rm_CA = [100, 1000]

    Model scaling - scales physical parameters to between 0-1 using min max.
    Physical scaling - returns parameters back to their physical meaning.

    Initialise with:
    InputScaling(C_rm, C1, C2, R1, R2, R3, Rin, cool, gain, solar)

    The energy (load) ranges are one row each, in LOAD_KEYS order: cool and gain in W/m2 of
    floor area, and solar as the dimensionless fraction p of global horizontal irradiance that
    reaches the room (solar gain = p * GHI * floor area). solar defaults to [0, 1] so older
    callers that only give cool and gain keep working.
    """

    def __init__(self, C_rm=None, C1=None, C2=None, R1=None, R2=None, R3=None, Rin=None, cool=None, gain=None, solar=(0, 1)):

        self.phys_param_range = [C_rm, C1, C2, R1, R2, R3, Rin]
        self.energy_param_range = [cool, gain, list(solar)]

        # check if the ranges are in the correct format. i.e. [lb, ub]
        for ranges in [self.phys_param_range, self.energy_param_range]:
            for bounds in ranges:
                assert bounds[0] <= bounds[1], f"Range for each parameter should be in the form [lb, ub], got: {bounds}"

    def physical_param_scaling(self, theta_scaled):
        """
        Scale from 0-1 back to physical value.
        """

        theta = self.unminmaxscale(theta_scaled, self.phys_param_range)

        return theta

    def model_param_scaling(self, theta):
        """
        Scale to 0-1.
        """

        theta_scaled = self.minmaxscale(theta, self.phys_param_range)

        return theta_scaled

    def physical_loads_scaling(self, loads_m):
        """
        Scale from 0-1 back to physical value in W/m2
        """
        self._check_load_rows(loads_m)
        loads_r = torch.zeros(loads_m.shape)
        for row, load_range in enumerate(self.energy_param_range):
            loads_r[row, :] = self.unminmaxscale(loads_m[row, :], load_range)

        return loads_r

    def model_loads_scaling(self, loads_r):
        """
        Scale to 0-1.
        """
        self._check_load_rows(loads_r)
        loads_m = torch.zeros(loads_r.shape)
        for row, load_range in enumerate(self.energy_param_range):
            loads_m[row, :] = self.minmaxscale(loads_r[row, :], load_range)

        return loads_m

    def _check_load_rows(self, loads):
        if loads.shape[0] != len(self.energy_param_range):
            raise ValueError(
                f"Expected one row of loads per energy range ({len(self.energy_param_range)}: cool, gain, solar), "
                f"got {loads.shape[0]}."
            )

    def minmaxscale(self, x, x_range):
        if not torch.is_tensor(x_range):
            x_range = torch.tensor(x_range)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x_range.ndim == 1:
            x_range = x_range.unsqueeze(0)

        # check if x too small (~ inverses)
        assert ~(x < x_range[:, 0]).any(), (
            f"From model inputs: {x}, input {x[(x < x_range[:, 0])]} is outside defined range: \n{x_range} "
        )
        # check if x too big
        assert ~(x > x_range[:, 1]).any(), (
            f"From model inputs: {x}, input {x[(x > x_range[:, 1])]} is outside defined range: \n{x_range} "
        )

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
