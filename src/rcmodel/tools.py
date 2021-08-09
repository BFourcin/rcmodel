from .RCModel import RCModel
import torch



class InputScaling(RCModel):
    """
    Class to group methods for scaling input parameters.
    Model scaling - scales physical parameters to between 0-1 using min max.
    Physical scaling - returns parameters back to their physical meaning.

    Initialise with:
    InputScaling(rm_CA ex_C, R, QA)
    or
    InputScaling(input_range=input_range)
    """

    def __init__(self, rm_CA=None, ex_C=None, R=None, Q_avg=None, input_range=None):

        if input_range is None:
            input_range = [rm_CA, ex_C, R, Q_avg]

            #check for None
            if None in input_range:
                raise ValueError('None type in list. Define all inputs or use a predefined input_range instead')


        self.input_range = torch.tensor(input_range)


    def physical_scaling(self, theta_scaled):
        """
        Scale from 0-1 back to normal.
        """

        rm_cap, ex_cap, ex_r, wl_r, Q_avg = self.categorise_theta(theta_scaled)

        
        rm_cap = self.unminmaxscale(rm_cap, self.input_range[0])
        ex_cap = self.unminmaxscale(ex_cap, self.input_range[1])
        ex_r   = self.unminmaxscale(ex_r  , self.input_range[2])
        wl_r   = self.unminmaxscale(wl_r  , self.input_range[2])
        Q_avg  = self.unminmaxscale(Q_avg , self.input_range[3])

        if wl_r.ndim == 0:
            wl_r = torch.unsqueeze(wl_r, 0)
        if Q_avg.ndim == 0:
            Q_avg = torch.unsqueeze(Q_avg, 0)

        theta = torch.cat([rm_cap, ex_cap, ex_r, wl_r, Q_avg])

        return theta


    def model_scaling(self, theta):
        """
        Scale to 0-1.
        """

        rm_cap, ex_cap, ex_r, wl_r, Q_avg = self.categorise_theta(theta)

        rm_cap = self.minmaxscale(rm_cap, self.input_range[0])
        ex_cap = self.minmaxscale(ex_cap, self.input_range[1])
        ex_r   = self.minmaxscale(ex_r  , self.input_range[2])
        wl_r   = self.minmaxscale(wl_r  , self.input_range[2])
        Q_avg  = self.minmaxscale(Q_avg , self.input_range[3])

        if wl_r.ndim == 0:
            wl_r = torch.unsqueeze(wl_r, 0)
        if Q_avg.ndim == 0:
            Q_avg = torch.unsqueeze(Q_avg, 0)

        theta_scaled = torch.cat([rm_cap, ex_cap, ex_r, wl_r, Q_avg])

        return theta_scaled


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
