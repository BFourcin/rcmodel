from ..physical import Room, Building
from ..rc_model import RCModel
from .rcmodel_dataset import BuildingTemperatureDataset
from xitorch.interpolate import Interp1D
from filelock import FileLock


import pandas as pd
import torch
import os


def initialise_model(pi, scaling, weather_data_path):

    def change_origin(coords):
        x0 = 92.07
        y0 = 125.94

        for i in range(len(coords)):
            coords[i][0] = round((coords[i][0] - x0) / 10, 2)
            coords[i][1] = round((coords[i][1] - y0) / 10, 2)

        return coords

    rooms = []

    name = "seminar_rm_a_t0106"
    coords = change_origin(
        [[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74], [172.64, 231.74], [172.64, 125.94]])
    rooms.append(Room(name, coords))

    # Initialise Building
    bld = Building(rooms)

    df = pd.read_csv(weather_data_path)
    Tout = torch.tensor(df['Hourly Temperature (°C)'])
    t = torch.tensor(df['time'])

    Tout_continuous = Interp1D(t, Tout, method='linear')

    # Initialise RCModel with the building
    transform = torch.sigmoid
    model = RCModel(bld, scaling, Tout_continuous, transform, pi)

    return model


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
            raise ValueError(
                'Data appears to have already been sorted. Check if still appropriate and add _sorted.csv tag to avoid this error.')

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

            else:  # File doesn't exist
                return True

    if need_to_sort(path, dt):
        sort(path, dt)

    # return the path_sorted
    if path[-11:] == '_sorted.csv':
        path_sorted = path
    else:
        path_sorted = path[:-4] + '_sorted.csv'

    return path_sorted


# def initialise_prior(scaling, weather_data_path):
#
#     # policy = PolicyNetwork(7,2)
#     prior = PriorCoolingPolicy()
#
#     model = initialise_model(prior, scaling, weather_data_path)
#
#     dt = 30
#     sample_size = 24 * 60 ** 2 / dt
#
#     op = OptimiseRC(model, csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)