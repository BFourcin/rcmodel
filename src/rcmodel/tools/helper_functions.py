from ..physical import Room, Building, InputScaling
from ..rc_model import RCModel
from ..optimisation import PolicyNetwork
from .rcmodel_dataset import BuildingTemperatureDataset, RandomSampleDataset
from xitorch.interpolate import Interp1D
from filelock import FileLock
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import os


def model_creator(model_config):
    # values taken from architectural drawing
    def change_origin(coords):
        x0 = 92.07
        y0 = 125.94

        for i in range(len(coords)):
            coords[i][0] = round((coords[i][0] - x0) / 10, 2)
            coords[i][1] = round((coords[i][1] - y0) / 10, 2)

        return coords

    def init_scaling():
        # Initialise scaling class
        C_rm = model_config['C_rm']  # [min, max] Capacitance/m2
        C1 = model_config['C1']  # Capacitance
        C2 = model_config['C2']
        R1 = model_config['R1']  # Resistance ((K.m^2)/W)
        R2 = model_config['R2']
        R3 = model_config['R3']
        Rin = model_config['Rin']
        cool = model_config['cool']  # Cooling limit in W/m2
        gain = model_config['gain']  # Gain limit in W/m2

        scaling = InputScaling(C_rm, C1, C2, R1, R2, R3, Rin, cool, gain)
        return scaling

    rooms = []
    for i in range(len(model_config['room_names'])):
        name = model_config['room_names'][i]
        coords = change_origin(model_config['room_coordinates'][i])
        rooms.append(Room(name, coords))

    # Initialise Building
    bld = Building(rooms)

    df = pd.read_csv(model_config['weather_data_path'])
    Tout = torch.tensor(df['Hourly Temperature (°C)'])
    t = torch.tensor(df['time'])

    Tout_continuous = Interp1D(t, Tout, method='linear')

    time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
    temp_data = torch.tensor(
        pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
        dtype=torch.float32)

    Tin_continuous = Interp1D(time_data, temp_data[:, 0:len(bld.rooms)].T, method='linear')

    pi = PolicyNetwork(5, 2)
    scaling = init_scaling()

    # Initialise RCModel with the building
    transform = torch.sigmoid
    model = RCModel(bld, scaling, Tout_continuous, Tin_continuous, transform, pi)

    # load physical and/or policy models if available
    if model_config['load_model_path_policy']:
        model.load(model_config['load_model_path_policy'])  # load policy
        model.init_physical()  # re-randomise physical params, as they were also copied from the loaded policy

    if model_config['load_model_path_physical']:
        # Try loading a dummy model with no policy, if it fails load with a policy. (We don't know what file contains)
        try:
            m = initialise_model(None, scaling, model_config['weather_data_path'], model_config['room_data_path'])
            m.load(model_config['load_model_path_physical'])

        except RuntimeError:
            m = initialise_model(pi, scaling, model_config['weather_data_path'], model_config['room_data_path'])
            m.load(model_config['load_model_path_physical'])

        model.params = m.params  # put loaded physical parameters onto model.
        model.loads = m.loads
        del m

    # check if any parameters have been chosen by the user:
    try:
        loads = model.loads.detach()
        if model_config['cooling_param']:
            loads[0, :] = torch.logit(torch.tensor(model_config['cooling_param']))

        if model_config['gain_param']:
            loads[1, :] = torch.logit(torch.tensor(model_config['gain_param']))

        model.loads = torch.nn.Parameter(loads)

    except KeyError as exception:  # input not found
        print(f'Exception during RCmodel creation, missing var in model config: {exception}')

    return model


def initialise_model(pi, scaling, weather_data_path, csv_path):
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

    time_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
    temp_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
                             dtype=torch.float32)

    Tin_continuous = Interp1D(time_data, temp_data[:, 0:len(bld.rooms)].T, method='linear')

    # Initialise RCModel with the building
    transform = torch.sigmoid
    model = RCModel(bld, scaling, Tout_continuous, Tin_continuous, transform, pi)

    return model


def dataset_creator(path, sample_size, dt):
    path_sorted = sort_data(path, dt)
    with FileLock(f"{os.path.dirname(os.path.abspath(path_sorted))}.lock"):
        # train_dataset = BuildingTemperatureDataset(path_sorted, sample_size, train=True)
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        # test_dataset = BuildingTemperatureDataset(path_sorted, sample_size, test=True)
        # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        warmup_size = 7 * sample_size
        train_dataset = RandomSampleDataset(path_sorted, sample_size, warmup_size, train=True, test=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_dataset = RandomSampleDataset(path_sorted, sample_size, warmup_size, train=False, test=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataset, train_dataloader, test_dataset, test_dataloader


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


def model_to_csv(observations, output_path):
    """
    Produces a .csv of the output. To then be used in return_to_sender.py
    """
    # Produce a .csv in the same format as current data, retains compatibility with dataloader
    titles = [['date-time', 'time'], [f'Rm{i}' for i in range(observations[:, 3:].shape[1])]]
    titles = [item for sublist in titles for item in sublist]

    # sort into date-time, unix time and temp data columns. (date-time exists just to keep format consistent)
    df = torch.hstack(
        (torch.zeros(len(observations), 1) * torch.nan, observations[:, 0].unsqueeze(1), observations[:, 3:]))
    df = pd.DataFrame(df.detach().numpy())
    df.to_csv(output_path, index=False, header=titles)


def convergence_criteria(y, n=10):
    """
    y: array of values.
    n: total lookback window size.

    Finds % difference between halves of a window of length n.
    Used as a measure of convergence.
    """

    if n % 2 != 0:
        raise TypeError('n must be even.')

    y_a = y[-n:-n // 2]
    y_b = y[-n // 2:]

    # formula doesn't work if there's insufficient data or with None
    # output will be 1 until y >= n
    if None in y_a or None in y_b:
        c = None

    elif len(y_a) == len(y_b):

        c = abs(sum(y_a) - sum(y_b)) / abs(sum(y_b))
    else:
        c = None

    return c


def exponential_smoothing(y, alpha, y_hat=None, n=10):
    # Check to see if y is an array, and we should calculate all values of y_hat
    # or if y is a single value, and therefore we just want the next value of y_hat
    try:
        if len(y) > 1:
            # y is an array meaning we want to calc y_hat for all values in array
            if y_hat:
                raise ValueError('Trying to smooth entire array, don\'t include y_hat')

            y_hat = []
            cycle = []
            index = 0
            for i, val in enumerate(y):  # get the non None parts of the list and find the convergence on them
                if i < index:
                    continue

                if val is None:
                    y_hat.append(None)
                    cycle = []  # reset
                else:
                    cycle.append(val)
                    if len(cycle) >= n:
                        y_hat.append(np.array(cycle[0:n]).mean())
                        index = i + 1
                        for yi in y[i + 1:]:
                            index += 1
                            if yi is None:
                                y_hat.append(None)
                                cycle = []  # reset
                                break
                            else:

                                y_hat.append(y_hat[-1] + alpha * (yi - y_hat[-1]))
                    else:
                        y_hat.append(None)

        else:
            # y is a list of len=1. supply a starting y_hat
            y_hat.append(y_hat[-1] + alpha * (y[0] - y_hat[-1]))

    except TypeError:
        # y is a int or float. supply a starting y_hat
        y_hat.append(y_hat[-1] + alpha * (y - y_hat[-1]))

    return y_hat


def policy_image(policy, n=100, path=None):
    bounds = [15, 30]
    t0 = 4 * 24 * 60 ** 2  # buffer to go from thursday to monday
    time = torch.linspace(0 + t0, 24 * 60 ** 2 + t0, n)
    temp = torch.linspace(bounds[0], bounds[1], n)
    img = torch.zeros((n, n))

    with torch.no_grad():
        for i, x in enumerate(temp):
            for j, y in enumerate(time):
                action, log_prob = policy.get_action(x.unsqueeze(0).unsqueeze(0), y)
                # Get prob of getting 1:
                if action == 1:
                    pr = torch.e ** log_prob  # Convert log_prob to normal prob.
                elif action == 0:
                    pr = 1 - torch.e ** log_prob  # pr(a=1) = 1 - pr(a=0)
                else:
                    raise ValueError(f'action={action}, must be exactly 1 or 0.')

                img[i, j] = pr

    fig = plt.figure()
    plt.imshow(img, origin='lower', aspect='auto', cmap='viridis', extent=(0, 24, bounds[0], bounds[1]), vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('Time of Day [hours]')
    plt.ylabel(r'Indoor Temperature [$^\circ$C]')
    plt.title('Policy Plot')
    plt.xticks(np.linspace(0, 24, 13))
    plt.yticks(np.linspace(bounds[0], bounds[1], 7))
    plt.grid(color='k', linestyle='--', )

    if path:
        fig.savefig(path)
    else:
        plt.show()
