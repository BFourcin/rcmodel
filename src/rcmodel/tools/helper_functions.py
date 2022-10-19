from ..physical import Room, Building, InputScaling
from ..rc_model import RCModel
from ..optimisation import PolicyNetwork
from .rcmodel_dataset import BuildingTemperatureDataset, RandomSampleDataset
from xitorch.interpolate import Interp1D
from torchdiffeq import odeint
from filelock import FileLock
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import os


def model_creator(model_config):
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

    pi = PolicyNetwork(5, 2)
    scaling = init_scaling()

    # Initialise RCModel with the building
    model = initialise_model(pi, scaling, model_config['weather_data_path'], model_config['room_names'], model_config['room_coordinates'])

    # load physical and/or policy models if available
    if model_config['load_model_path_policy']:
        model.load(model_config['load_model_path_policy'])  # load policy
        model.init_physical()  # re-randomise physical params, as they were also copied from the loaded policy

    if model_config['load_model_path_physical']:
        # Try loading a dummy model with no policy, if it fails load with a policy. (We don't know what file contains)
        try:
            m = initialise_model(None, scaling, model_config['weather_data_path'], model_config['room_names'], model_config['room_coordinates'])
            m.load(model_config['load_model_path_physical'])

        except RuntimeError:
            m = initialise_model(pi, scaling, model_config['weather_data_path'], model_config['room_names'], model_config['room_coordinates'])
            m.load(model_config['load_model_path_physical'])

        model.params = m.params  # put loaded physical parameters onto model.
        model.loads = m.loads
        del m

    # check if any parameters have been chosen by the user:
    try:
        loads = model.loads.detach()
        if model_config['parameters'] is not None:

            parameters = []
            for param in model_config['parameters']:
                parameters.append(model_config['parameters'][param])

            assert len(parameters) == 9, 'Include all parameters'

            params = torch.logit(torch.tensor(parameters[0:7]))
            loads = torch.logit(torch.tensor(parameters[7:]).unsqueeze(0).T)

        model.params = torch.nn.Parameter(params)
        model.loads = torch.nn.Parameter(loads)

    except KeyError as exception:  # input not found
        print(f'Exception during RCmodel creation, missing var in model config: {exception}')

    return model


def initialise_model(pi, scaling, weather_data_path, room_names, room_coordinates):
    def change_origin(coords):
        """This function changes the origin of the coordinate system, so [0,0] is on the building.
        Function exists because floor plan was offset."""
        x0 = 92.07
        y0 = 125.94

        new_coords = []
        for i in range(len(coords)):
            l = [round((coords[i][0] - x0) / 10, 2), round((coords[i][1] - y0) / 10, 2)]
            new_coords.append(l)

        return new_coords

    rooms = []
    for i in range(len(room_names)):
        name = room_names[i]
        coords = change_origin(room_coordinates[i])
        rooms.append(Room(name, coords))

    # Initialise Building
    bld = Building(rooms)

    df = pd.read_csv(weather_data_path)
    Tout = torch.tensor(df['Hourly Temperature (°C)'])
    t = torch.tensor(df['time'])
    Tout_continuous = Interp1D(t, Tout, method='linear')  # Interp1D object

    # Initialise RCModel with the building
    transform = torch.sigmoid
    model = RCModel(bld, scaling, Tout_continuous, transform, pi)

    return model


def dataloader_creator(path, sample_size, warmup_size, dt=30):
    path_sorted = sort_data(path, dt)
    with FileLock(f"{os.path.dirname(os.path.abspath(path_sorted))}.lock"):
        # train_dataset = BuildingTemperatureDataset(path_sorted, sample_size, train=True)
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        # test_dataset = BuildingTemperatureDataset(path_sorted, sample_size, test=True)
        # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_dataset = RandomSampleDataset(path_sorted, sample_size, warmup_size, train=True, test=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_dataset = RandomSampleDataset(path_sorted, sample_size, warmup_size, train=False, test=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

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
        infer_dst = np.array([False] * df.shape[
            0])  # all False -> every row considered DT, alternative is True to indicate DST. The array must correspond to the iloc of df.index
        df = df.tz_localize('Europe/London', ambiguous=infer_dst,
                            nonexistent='shift_forward')  # causes error so commented out

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


def get_iv_array(model, dataset):
    """
    Perform the integration but force outside and inside temperature to be from data.
    Only the latent temperature nodes in the external walls are free to change meaning we can find out their true
    values for a given model.
    """
    # Update building with current parameters, if needed.
    if not torch.eq(model.params_old, model.params).all():
        model._build_matrices()

    with torch.no_grad():
        t_eval, temp_data = dataset.get_all_data()
        Tin_continuous = Interp1D(t_eval, temp_data[:, 0:len(model.building.rooms)].T, method='linear')

        bl = model.building

        # Recalculate the A & B matrices. We could chop and re jig from the full matrices, but it is not super
        # simple so recalculating is less risky.
        A = torch.zeros([2, 2])
        A[0, 0] = bl.surf_area * (-1 / (bl.Re[0] * bl.Ce[0]) - 1 / (bl.Re[1] * bl.Ce[0]))
        A[0, 1] = bl.surf_area / (bl.Re[1] * bl.Ce[0])
        A[1, 0] = bl.surf_area / (bl.Re[1] * bl.Ce[1])
        A[1, 1] = bl.surf_area * (-1 / (bl.Re[1] * bl.Ce[1]) - 1 / (bl.Re[2] * bl.Ce[1]))

        B = torch.zeros([2, 2])
        B[0, 0] = bl.surf_area / (bl.Re[0] * bl.Ce[0])
        B[1, 1] = bl.surf_area / (bl.Re[2] * bl.Ce[1])

        # check if t_eval is formatted correctly:
        if t_eval.dim() > 1:
            t_eval = t_eval.squeeze(0)

        avg_tout = model.Tout_continuous(t_eval).mean()
        avg_tin = Tin_continuous(t_eval).mean()

        t0 = t_eval[0]
        t_eval = t_eval - t0

        model.iv = steady_state_iv(model, avg_tout, avg_tin)  # Use avg temp as a good starting guess for iv.

        def latent_f_ode(t, x):
            Tout = model.Tout_continuous(t.item() + t0)
            Tin = Tin_continuous(t.item() + t0)

            u = torch.tensor([[Tout],
                              [Tin]])

            return A @ x + B @ u.to(torch.float32)

        integrate = odeint(latent_f_ode, model.iv[0:2], t_eval,
                           method='rk4')  # https://github.com/rtqichen/torchdiffeq

        integrate = integrate.squeeze()

        # Add on inside temperature data to be used to initialise rooms at the correct temp.
        iv_array = torch.empty(len(integrate), len(bl.rooms) + 2)
        iv_array[:, 0:2] = integrate
        iv_array[:, 2:] = Tin_continuous(t_eval + t0).T
        iv_array = Interp1D(t_eval + t0, iv_array.T, method='linear')

    return iv_array


def steady_state_iv(model, temp_out, temp_in):
    """
    Calculate the initial conditions of the latent variables given a steady state indoor and outdoor temperature.
    Initial values of room nodes are set to temp in.

    temp_out: float
        Steady state outside temperature.
    temp_in: tensor
        Steady state inside temperature. len(temp_in) = n_rooms
    :return: tensor
        Column tensor of initial values at each node.
    """
    I = (temp_out - temp_in) / sum(model.building.Re)  # I=V/R
    v1 = temp_out - I * model.building.Re[0]
    v2 = v1 - I * model.building.Re[1]

    iv = torch.tensor([[v1], [v2], [temp_in]]).to(torch.float32)

    return iv