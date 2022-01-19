from .building import Building
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt


class InputScaling(Building):
    """
    Class to group methods for scaling input parameters.
    Provide the limits for each variable. e.g. rm_CA = [100, 1000]

    Model scaling - scales physical parameters to between 0-1 using min max.
    Physical scaling - returns parameters back to their physical meaning.

    Initialise with:
    InputScaling(rm_CA ex_C, R, Q_limit)
    or
    InputScaling(input_range=input_range)
    """

    def __init__(self, rm_CA=None, ex_C=None, R=None, Q_limit=None, input_range=None):

        if input_range is None:
            input_range = [rm_CA, ex_C, R, Q_limit]

            # check for None
            if None in input_range:
                raise ValueError('None type in list. Define all inputs or use a predefined input_range instead')

        self.input_range = torch.tensor(input_range)

    def physical_param_scaling(self, theta_scaled):
        """
        Scale from 0-1 back to normal.
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

        Q_watts = self.unminmaxscale(Q, [0, self.input_range[3].max()])

        return Q_watts

    def model_cooling_scaling(self, Q_watts):

        Q = self.minmaxscale(Q_watts, [0, self.input_range[3].max()])

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


class BuildingTemperatureDataset(Dataset):
    """
    Splits dataset up into batches of len(dataset) // sample_size. Note remainder of data is thrown away.
    train and test tags can be used to select a percentage slice of data in this order, see function _split_dataset() for current % splits.

    If there is insufficient data for one batch, sample_size will be reduced to match the data.
    """
    def __init__(self, csv_path, sample_size, transform=None, all=True, train=False, test=False):
        self.csv_path = csv_path
        self.transform = transform
        self.sample_size = int(sample_size)
        self.headings = list(pd.read_csv(csv_path, nrows=1)) # list of dataframe headings
        self.all = all
        self.train = train
        self.test = test

        # auto splits data by train and test
        # entry count is number of rows to read from csv.
        # remainder of data e.g. after entry_count//sample_size is lost
        self.rows_to_skip, self.entry_count = self._split_dataset()

        self.__len__()  # Used to initialise logic used in __len__.

    def __len__(self):
        """
        Get number of batches in the dataset. Returns int
        Minimum of 1 batch will be returned.
        """

        num_samples = self.entry_count//self.sample_size

        # Insufficient data for 1 whole sample size. Remainder of data used instead.
        if num_samples == 0:
            num_samples = 1
            self.sample_size = self.entry_count  # Reduce sample to the entries remaining

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

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64) # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(df_sample.iloc[:, 2:].values, dtype=torch.float32) #pandas needs 2: to get all but first & second column

        #apply transorms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def _get_entries(self):
        """Get total rows/entries of data in csv"""
        from csv import reader

        # the rows in the .csv are counted:
        with open(self.csv_path,"r") as f:
            read_f = reader(f,delimiter = ",")
            entry_count = sum(1 for row in read_f) - 1 # minus one to account for heading

        return entry_count

    def _split_dataset(self):

        train_split = 0.8
        test_split = 0.2

        total_entries = self._get_entries() #total rows of data in csv

        if self.train:
            rows_to_skip = 0
            entry_count = int(total_entries*train_split)
            return rows_to_skip, entry_count
        elif self.test:
            rows_to_skip = int(total_entries*train_split)
            entry_count = int(total_entries*test_split)
            return rows_to_skip, entry_count
        elif self.all:
            rows_to_skip = 0
            entry_count = total_entries
            return rows_to_skip, entry_count
        else:
            raise ValueError('train, test and validation all False')


# helper functions:
def pltsolution_1rm(model, dataloader, filename=None):
    """
    Plots the first sample of dataloader.
    """
    model.reset_iv()  # Reset initial value
    model.eval()  # Put model in evaluation mode

    # Get solution ---------------
    time, data_temp = next(iter(dataloader))

    time = time.squeeze(0)
    data_temp = data_temp.squeeze(0)

    t_days = (time - time.min()) / (24 * 60 ** 2)  # Time in days

    pred = model(time)
    pred = pred.squeeze(-1)

    # Get Heating Control for each room
    if model.cooling_policy:
        record_action = torch.tensor(model.record_action)
        Q_tdays = record_action[:, 0] / (24 * 60 ** 2)  # Time in days
        Q_on_off = record_action[:, 1:]  # Cooling actions

        Q_avg = model.transform(model.cooling)
        Q_avg = model.scaling.physical_cooling_scaling(Q_avg)

        Q = Q_on_off * -Q_avg.unsqueeze(1)  # Q is timeseries of Watts for each room.
    else:
        Q_tdays = t_days
        Q = torch.zeros(len(Q_tdays))

    # Compute and print loss.
    # loss_fn = torch.nn.MSELoss()
    # num_cols = len(model.building.rooms)  # number of columns to take from data.
    # loss = loss_fn(pred[:, 2:], data_temp[:, 0:num_cols])

    # print(f"Test loss = {loss.item():>8f}")

    # ---------------------------------

    # Plot Solution

    ax2ylim = 250

    fig, axs = plt.subplots(figsize=(10, 8))

    ax2 = axs.twinx()
    ln1 = axs.plot(t_days.detach().numpy(), pred[:, 2:].detach().numpy(), label='model')
    ln2 = axs.plot(t_days.detach().numpy(), data_temp[:, 0].detach().numpy(), label='data')
    ln3 = axs.plot(t_days.detach().numpy(), model.Tout_continuous(time).detach().numpy(), label='outside')
    ln4 = ax2.plot(Q_tdays.detach().numpy(), Q.detach().numpy(), '--', color='black', alpha=0.5, label='heat')
    axs.set_title(model.building.rooms[0].name)
    ax2.set_ylabel(r"Heating/Cooling ($W/m^2$)")
    # ax2.set_ylim(-ax2ylim, ax2ylim)

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    axs.legend(lns, labs, loc=0)

    axs.set(xlabel='Time (days)', ylabel='Temperature ($^\circ$C)')

    if filename:
        fig.savefig(filename)
        plt.close()

    else:
        plt.show()