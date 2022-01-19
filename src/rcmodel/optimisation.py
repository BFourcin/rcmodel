from filelock import FileLock
import torch
import pandas as pd
from .tools import BuildingTemperatureDataset
import os


def train(model, dataloader, optimizer):
    """
    Performs one epoch of training.
    Order of rooms in building and in data must match otherwise model will fit wrong rooms to data.
    """
    model.reset_iv()  # Reset initial value
    model.train()
    model.cooling_policy.eval()

    # Stops Autograd endlessly keeping track of the graph. Memory Leak!
    for layer in model.cooling_policy.parameters():
        layer.requires_grad = False

    num_cols = len(model.building.rooms)  # number of columns to use from data.
    num_batches = len(dataloader)
    train_loss = 0
    loss_fn = torch.nn.MSELoss()

    for batch, (time, temp) in enumerate(dataloader):
        # Get model arguments:
        time = time.squeeze(0)
        temp = temp.squeeze(0)

        # Compute prediction and loss
        pred = model(time)
        pred = pred.squeeze(-1)  # change from column to row matrix

        loss = loss_fn(pred[:, 2:], temp[:, 0:num_cols])
        train_loss += loss.item()

        # get last output and use for next initial value
        model.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / num_batches


def test(model, dataloader):
    model.reset_iv()  # Reset initial value
    model.eval()  # Put model in evaluation mode
    num_batches = len(dataloader)
    num_cols = len(model.building.rooms)  # number of columns to take from data.
    test_loss = 0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for (time, temp) in dataloader:

            time = time.squeeze(0)
            temp = temp.squeeze(0)

            pred = model(time)
            pred = pred.squeeze(-1)  # change from column to row matrix
            test_loss += loss_fn(pred[:, 2:], temp[:, 0:num_cols]).item()

            # get last output and use for next initial value
            model.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

    test_loss /= num_batches

    return test_loss


def dataset_creator(path, sample_size, dt):
    path_sorted = sort_data(path, dt)
    with FileLock(f"{os.path.dirname(os.path.abspath(path_sorted))}.lock"):
        training_data = BuildingTemperatureDataset(path_sorted, sample_size, train=True)
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)
        test_data = BuildingTemperatureDataset(path_sorted, sample_size, test=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        
    return train_dataloader, test_dataloader


def sort_data(path, dt):
    """
    Check if path has sorted data tag (_sorted)
    If not check if data has previously been sorted and exists in the directory.
    Check to see if the value dt is correct
    If not sort data and write filename_sorted.csv

    data is sorted by time in ascending order and downsampled to a frequency of dt seconds.
    Missing values are interpolated.
    A time-date string is also inserted.
    """
    def sort(path, dt):
        df = pd.read_csv(path)

        if path[-11:] == '_sorted.csv':
            path_sorted = path
        else:
            path_sorted = path[:-4] + '_sorted.csv'

        # Sort df by time (raw data not always in order)
        df = df.sort_values(by=["time"], ascending=True)

        # insert date-time value at start of df
        try:
            df.insert(loc=0, column='date-time', value=pd.to_datetime(df['time'], unit='ms'))
        except ValueError:
            raise ValueError('Data appears to have already been sorted. Check if still appropriate and add _sorted.csv tag to avoid this error.')

        # downscale data to a frequency of dt (seconds) use the mean value and round to 2dp.
        df = df.set_index('date-time').resample(str(dt) + 's').mean().round(2)

        # time column is converted to unix epoch seconds to match the date-time
        df["time"] = (df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        # change date-time from UTC to Local time
        df = df.tz_localize('Europe/London')

        df = df.interpolate().round(2)  # interpolate missing values NaN

        df.to_csv(path_sorted, index=True)

    def need_to_sort(path, dt):

        def get_dt(path):
            df_dt = pd.read_csv(path)['time'][0:2].values
            return df_dt[1] - df_dt[0]

        # Does path already have sorted tag?
        if path[-11:] == '_sorted.csv':

            # if so, is dt correct?
            if get_dt(path) == dt:

                return False  # path and file is correct dont sort

            else:
                return True  # dt is wrong, re-sort

        # path does not contain _sorted.csv
        else:

            # Does path_sorted exist?
            path_sorted = path[:-4] + '_sorted.csv'
            import os.path
            if os.path.isfile(path_sorted):  # check if file already exists

                # if file exists check if dt is correct
                if get_dt(path_sorted) == dt:
                    return False  # correct file already exists don't sort
                else:
                    return True  # file exists but dt wrong, re-sort

            else: # File doesn't exist
                return True

    if need_to_sort(path, dt):
        sort(path, dt)

    # return the path_sorted
    if path[-11:] == '_sorted.csv':
        path_sorted = path
    else:
        path_sorted = path[:-4] + '_sorted.csv'

    return path_sorted


class OptimiseRC:
    """
    Parameters
    ----------
    model : object
        RCModel class object.
    csv_path : string
        Path to .csv file containing room temperature data.
        Data will be sorted if not done already and saved to a new file with the tag '_sorted'
    sample_size : int
        Length of indexes to sample from dataset per batch.
    dt : int
        Timestep data will be resampled to.
    lr : float
        Learning rate for optimiser.
    model_id : int
        Unique identifier used when optimising multiple models.

    see https://docs.ray.io/en/latest/using-ray-with-pytorch.html
    """
    def __init__(self, model, csv_path, sample_size, dt=30, lr=1e-3, opt_id=0):
        self.model = model
        self.model.init_params()  # randomise parameters
        self.model_id = opt_id
        self.train_dataloader, self.test_dataloader = dataset_creator(csv_path, int(sample_size), int(dt))

        self.optimizer = torch.optim.Adam([self.model.params, self.model.cooling], lr=lr)



    def train(self):
        avg_loss = train(self.model, self.train_dataloader, self.optimizer)
        return avg_loss

    def test(self):
        test_loss = test(self.model, self.test_dataloader)
        return test_loss

    def train_loop(self, epochs):
        print(self.model.params)
        for i in range(int(epochs)):
            # print(f"Epoch {i + 1}\n-------------------------------")
            testloss = self.train()

        results = [testloss, self.model]
        return results











