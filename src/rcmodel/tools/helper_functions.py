from rcmodel.physical import Room, Building, InputScaling
from rcmodel.rc_model import RCModel
import rcmodel.optimisation
from .rcmodel_dataset import BuildingTemperatureDataset, RandomSampleDataset
from gymnasium.wrappers import RenderCollection
from xitorch.interpolate import Interp1D
from torchdiffeq import odeint
from filelock import FileLock
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import os


def model_creator(model_config):
    """
    model_config = {
        # Ranges:
        "C_rm": [1e3, 1e5],  # [min, max] Capacitance/m2
        "C1": [1e5, 1e8],  # Capacitance
        "C2": [1e5, 1e8],
        "R1": [0.1, 5],  # Resistance ((K.m^2)/W)
        "R2": [0.1, 5],
        "R3": [0.5, 6],
        "Rin": [0.1, 5],
        "cool": [0, 50],  # Cooling limit in W/m2
        "gain": [0, 5],  # Gain limit in W/m2
        "room_names": ["seminar_rm_a_t0106"],
        "room_coordinates": [[[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74],
                              [172.64, 231.74], [172.64, 125.94]]],
        "weather_data_path": weather_data_path,
        "cooling_policy": None,
        "load_model_path_policy": None,  # './prior_policy.pt',  # or None
        "load_model_path_physical": None,  # or None
        "parameters": {
            "C_rm": np.random.rand(1).item(),
            "C1": np.random.rand(1).item(),
            "C2": np.random.rand(1).item(),
            "R1": np.random.rand(1).item(),
            "R2": np.random.rand(1).item(),
            "R3": np.random.rand(1).item(),
            "Rin": np.random.rand(1).item(),
            "cool": np.random.rand(1).item(),  # 0.09133423646610082
            "gain": np.random.rand(1).item(),  # 0.9086668150306394
        }
    }
    """
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

    # pi = PolicyNetwork(5, 2)
    pi = model_config["cooling_policy"]
    scaling = init_scaling()

    # Initialise RCModel with the building
    model = initialise_model(pi, scaling, model_config['weather_data_path'], model_config['room_names'],
                             model_config['room_coordinates'])

    # load physical and/or policy models if available
    if model_config['load_model_path_policy']:
        model.load(model_config['load_model_path_policy'])  # load policy
        model.initialise_parameters()  # re-randomise physical params, as they were also copied from the loaded policy

    if model_config['load_model_path_physical']:
        # Try loading a dummy model with no policy, if it fails load with a policy. (We don't know what file contains)
        try:
            m = initialise_model(None, scaling, model_config['weather_data_path'], model_config['room_names'],
                                 model_config['room_coordinates'])
            m.load(model_config['load_model_path_physical'])

        except RuntimeError:
            m = initialise_model(pi, scaling, model_config['weather_data_path'], model_config['room_names'],
                                 model_config['room_coordinates'])
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


def env_creator(env_config):
    """
    Creates a Reinforcement Learning environment for use with the Ray RLlib library.

    If an existing RCModel object is provided in env_config, it will be used directly. If not, the function
    will attempt to load an RCModel object from a pickled file path provided in env_config. If that fails,
    it will create a new RCModel object using the model configuration specified in env_config. If an RCModel
    object is successfully obtained or created, its state_dict can be optionally loaded from a dictionary
    provided in env_config.

    Args:
        env_config (dict): A dictionary containing the configuration settings for the environment. It can have
            the following keys:
            - "RC_model": An optional RCModel object that will be used directly if provided.
            - "model_pickle_path": An optional path to a pickled RCModel object.
            - "model_config": Required if RC_model and model_pickle_path are not provided. A dictionary containing
                the configuration settings for the RCModel object to be created. It should be in the same format
                as the config dictionaries used to create RCModel objects.
            - "update_state_dict": An optional dictionary containing the state_dict for the RCModel object. If provided,
                it will be loaded into the RCModel object before it is used to create the environment.
            - "dataloader": Required. A PyTorch DataLoader object that will be used to provide input data to the
                RCModel object. It should be configured to return batches of data in the format expected by the
                RCModel object.
            - "step_length": Required. The number of minutes passed in each environment step.
            - "render_mode": Optional. The render mode to use for the environment. Currently, only "single_rgb_array"
                is supported.

    Returns:
        env (gym.Env): A Reinforcement Learning environment that is ready for use with RLlib.
    """
    with torch.no_grad():

        # Try to get the model from env_config
        model = env_config.get("RC_model", None)

        # If model is not provided, try to load it from a pickle file
        if model is None:
            model_pickle_path = env_config.get("model_pickle_path", None)
            if model_pickle_path:
                model = RCModel.load(model_pickle_path)

        # If model is still not available, create one from a provided model_config.
        if model is None:
            model = model_creator(env_config["model_config"])

        # Finally, check if update_state_dict has been provided and load it if so.
        update_state_dict = env_config.get("update_state_dict", None)
        if update_state_dict:
            model.load_state_dict(update_state_dict)
        env_config["update_state_dict"] = None  # We've done the update.

        # env_config now has a model and can be used to create an environment.
        env_config["RC_model"] = model

        # Let's make a new config of just the items needed for the environment
        env_keys = ["RC_model", "dataloader", "step_length", "render_mode",
                    "update_state_dict"]
        config = {}
        for key in env_keys:
            config[key] = env_config[key]

        env = rcmodel.optimisation.LSIEnv(config)

        # wrap environment:
        env = rcmodel.optimisation.PreprocessEnv(env, mu=23.359, std_dev=1.41)

        # Wrap with nice render list api if we want get renders.
        if env_config["render_mode"] is not None:
            env = RenderCollection(env)
    return env


def env_create_and_setup(env_config):
    """
    Call env_creator and then set up RC model with system matrix and get iv_array.
    Usage with:
        register_env("LSIEnv", env_create_and_setup)

    Parameters
    ----------
    env_config: dict
        Configuration for environment.

    Returns
    -------
    env: gym.Env
        LSIEnv environment with RC model set up.
    """

    env = env_creator(env_config)
    # Rebuild matrices and get iv_array, needed after parameter change
    env.RC.setup(env.dataloader.dataset)
    return env


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


def policy_image(algo, n=100, path=None):
    """

    :param algo: ray RRLIB algo

    """
    import rcmodel
    bounds = [15, 30]
    t0 = 4 * 24 * 60 ** 2  # buffer to go from thursday to monday
    time = torch.linspace(0 + t0, 24 * 60 ** 2 + t0, n)
    temp = torch.linspace(bounds[0], bounds[1], n)
    img = torch.zeros((n, n))

    # hardcode mu and std_dev:
    mu = 23.359
    std_dev = 1.41

    with torch.no_grad():
        for i, te in enumerate(temp):
            for j, ti in enumerate(time):
                unix_time = ti
                x = te.unsqueeze(0)  # remove the latent nodes
                observation = rcmodel.optimisation.preprocess_observation(x, unix_time, mu, std_dev)
                action, _, info = algo.compute_action(observation, full_fetch=True)
                log_prob = info['action_logp']
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

