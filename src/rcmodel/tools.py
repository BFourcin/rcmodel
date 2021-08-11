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



from torch.utils.data import Dataset
import pandas as pd

class BuildingTemperatureDataset(Dataset):
    def __init__(self, csv_path, sample_size, rows_to_skip=0, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.sample_size = sample_size
        self.headings = list(pd.read_csv(csv_path, nrows=1)) # list of dataframe headings
        self.rows_to_skip = rows_to_skip
        self.entry_count = self._get_entries()

    def __len__(self):
        """Get number of batches in the dataset. Returns int"""


        # Find how many samples we get from dataset
#         if entry_count%self.sample_size == 0:
#             num_samples = int(entry_count/self.sample_size)
#         else:
#             num_samples = entry_count//self.sample_size + 1


        num_samples = self.entry_count//self.sample_size #clean this up later. Remainder of data is now lost.

        return num_samples

    def __getitem__(self, idx):
        """
        Returns a 'sample' of the total dataset.

        Each sample has length 'sample_size' except for the final one, which contains the remainder of the dataset.

        The entire dataset is never loaded in at once, each sample is read in separately when needed to reduce memory usage.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get upper and lower bounds of dataset slice
        lb = idx*self.sample_size + self.rows_to_skip
        ub = (idx+1)*self.sample_size + self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=lb, nrows=self.sample_size)

        #convert to tensor
        sample = torch.tensor(df_sample.values)

        #apply transorms if required
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_entries(self):
        """Get total rows/entries in dataset"""
        from csv import reader

        # the rows in the .csv are counted:
        with open(self.csv_path,"r") as f:
            read_f = reader(f,delimiter = ",")
            entry_count = sum(1 for row in read_f) - 1 # minus one to account for heading

        return entry_count - self.rows_to_skip
