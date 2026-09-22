import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from gymnasium.wrappers import RenderCollection
from xitorch.interpolate import Interp1D

import rcmodel.optimisation
from rcmodel.physical import Building, InputScaling, Room
from rcmodel.rc_model import RC_PARAM_KEYS, RCModel

from .rcmodel_dataset import BuildingTemperatureDataset, InfiniteSampler, RandomSampleDataset

# Normalisation constants for PreprocessEnv. These were hardcoded at the point of use; they
# are defaults now so a different building can override them via env_config without editing
# the library. They should really be computed from the training split - see make_dataloaders.
DEFAULT_OBSERVATION_MU = 23.359
DEFAULT_OBSERVATION_STD_DEV = 1.41


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
        "weather_data_outdoor_temperature": DataFrame or Array like,
        "weather_data_UTC_time": DataFrame or Array like,
        "cooling_policy": None,
        "load_model_path_policy": None,  # './prior_policy.pt',  # or None
        "load_model_path_physical": None,  # or None
        "parameters": {  # scaled 0-1, or None to initialise randomly
            "C_rm": np.random.rand(1),
            "C1": np.random.rand(1),
            "C2": np.random.rand(1),
            "R1": np.random.rand(1),
            "R2": np.random.rand(1),
            "R3": np.random.rand(1),
            "Rin": np.random.rand(1),
            "cool": np.random.rand(len(room_coordinates)),  # one per room, or a single value for all
            "gain": np.random.rand(len(room_coordinates)),
        }
    }
    """
    # Names of the [min, max] ranges. The "parameters" dict uses the same names.
    range_keys = RC_PARAM_KEYS

    def init_scaling():
        # Initialise scaling class
        C_rm = model_config["C_rm"]  # [min, max] Capacitance/m2
        C1 = model_config["C1"]  # Capacitance
        C2 = model_config["C2"]
        R1 = model_config["R1"]  # Resistance ((K.m^2)/W)
        R2 = model_config["R2"]
        R3 = model_config["R3"]
        Rin = model_config["Rin"]
        cool = model_config["cool"]  # Cooling limit per room in W/m2
        gain = model_config["gain"]  # Gain limit per room in W/m2

        scaling = InputScaling(C_rm, C1, C2, R1, R2, R3, Rin, cool, gain)
        return scaling

    def model_sanity_checks():
        """Check the config is self-consistent before anything is built."""
        n_weather = len(model_config["weather_data_outdoor_temperature"])
        n_time = len(model_config["weather_data_UTC_time"])
        assert n_weather == n_time, (
            f"Weather data length mismatch: 'weather_data_outdoor_temperature' has {n_weather} points, "
            f"'weather_data_UTC_time' has {n_time}."
        )

        n_names = len(model_config["room_names"])
        n_rooms = len(model_config["room_coordinates"])
        assert n_names == n_rooms, f"Each room needs a name: got {n_names} names for {n_rooms} rooms."

        # InputScaling checks min <= max, here we only check the shape of each range.
        for key in range_keys:
            assert len(model_config[key]) == 2, f"Range for '{key}' should be [min, max], got: {model_config[key]}"

        # Starting parameters are optional, but if given they must be complete.
        # Their lengths are checked where the tensors are built, as a single value is allowed for every room.
        if model_config.get("parameters") is not None:
            missing = [key for key in range_keys if key not in model_config["parameters"]]
            assert not missing, f"model_config['parameters'] is missing: {missing}"

        return

    model_sanity_checks()

    pi = model_config.get("cooling_policy")
    scaling = init_scaling()

    # Initialise RCModel with the building
    model = initialise_model(
        pi,
        scaling,
        model_config["weather_data_outdoor_temperature"],
        model_config["weather_data_UTC_time"],
        model_config["room_names"],
        model_config["room_coordinates"],
    )

    # NOTE: the old "load_model_path_policy" branch has been removed. RCModel.load is a
    # staticmethod, so `model.load(path)` built a model and threw it away - the branch only
    # ever re-randomised the parameters. The cooling policy now lives in the RLlib
    # checkpoint, not inside the RCModel, so there is nothing for it to load.

    if model_config.get("load_model_path_physical"):
        # NOTE: this used to build a throwaway model and call m.load(path) on it. RCModel.load
        # is a staticmethod, so that returned a new model which was discarded, and the
        # parameters copied out of `m` afterwards were its freshly randomised ones - the
        # branch never loaded anything. Use the returned model.
        loaded = RCModel.load(model_config["load_model_path_physical"])

        # A model pickled before parameters moved to 0-1 machine space stored them in logit
        # space with a sigmoid transform. Apply that model's own transform (if it had one) so
        # an older file still loads to the right physical values.
        loaded_params = loaded.transform(loaded.params) if loaded.transform else loaded.params
        loaded_loads = loaded.transform(loaded.loads) if loaded.transform else loaded.loads

        model.params = torch.nn.Parameter(loaded_params.detach().clone(), requires_grad=False)
        model.loads = torch.nn.Parameter(loaded_loads.detach().clone(), requires_grad=False)
        del loaded

    # check if any parameters have been chosen by the user:
    if model_config.get("parameters") is not None:
        # Parameters are stored directly in 0-1 machine space. They used to be pushed
        # through torch.logit here to undo the model's sigmoid transform, which only
        # existed to keep gradient descent on unbounded parameters well behaved. There is
        # no gradient descent any more, and logit(0)/logit(1) are -/+inf, which a search
        # that can land on a range endpoint would hit. set_parameters() checks the values
        # are physically valid (not that they sit inside the configured range) and handles
        # the per-room broadcasting of cool/gain.
        model.set_parameters(model_config["parameters"])

    return model


def env_creator(env_config):
    """
    Creates a Reinforcement Learning environment for use with the Ray RLlib library.

    The RC model is obtained in the first of these ways that works:
      1. env_config["RC_model"], an already-built RCModel (handy in tests).
      2. env_config["model_pickle_path"], a pickled RCModel.
      3. env_config["model_config"], a config dict - see model_creator().

    For a PBT run, use route 3. The trial's sampled RC parameters live in
    model_config["parameters"] (0-1 scaled), so the environment is fully described by plain
    data in the config, and a trial's parameters are whatever its config says they are.
    Route 1 puts a live, mutable object in the config, which Tune would deep-copy between
    trials - parameters would stop tracking the config and a PBT exploit would silently do
    nothing.

    Note this function does NOT write back into env_config. It used to stash the constructed
    model under "RC_model", which had exactly the aliasing problem described above.

    Args:
        env_config (dict): Configuration for the environment. Keys:
            - "RC_model" / "model_pickle_path" / "model_config": the model source, see above.
            - "dataloader": Required unless "data_config" is given. Provides episode data.
            - "data_config": Alternative to "dataloader" - see make_dataloaders(). Keeps the
                config plain data, which is what a distributed search wants.
            - "step_length": Required. Minutes of data per environment step.
            - "render_mode": Optional.
            - "rc_parameters": Optional dict of 0-1 scaled parameters applied on top of the
                model, the route a PBT exploit uses.
            - "update_state_dict": Optional RCModel state_dict applied on top of the model.
            - "observation_mu" / "observation_std_dev": Optional normalisation constants.

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

        dataloader = env_config.get("dataloader")
        if dataloader is None:
            dataloader, _ = make_dataloaders(env_config["data_config"])

        # Build the environment's own config rather than handing it the caller's dict.
        config = {
            "RC_model": model,
            "dataloader": dataloader,
            "step_length": env_config["step_length"],
            "render_mode": env_config.get("render_mode"),
            "rc_parameters": env_config.get("rc_parameters"),
            "update_state_dict": env_config.get("update_state_dict"),
        }

        env = rcmodel.optimisation.LSIEnv(config)

        # wrap environment:
        env = rcmodel.optimisation.PreprocessEnv(
            env,
            mu=env_config.get("observation_mu", DEFAULT_OBSERVATION_MU),
            std_dev=env_config.get("observation_std_dev", DEFAULT_OBSERVATION_STD_DEV),
        )

        # Wrap with nice render list api if we want get renders.
        if config["render_mode"] is not None:
            env = RenderCollection(env)
    return env


def env_create_and_setup(env_config):
    """
    Deprecated alias for env_creator().

    LSIEnv now builds its own A/B matrices and iv_array in __init__, so there is no longer a
    separate setup step to forget. Kept so existing register_env("LSIEnv", ...) calls keep
    working.
    """
    return env_creator(env_config)


def make_dataloaders(data_config):
    """
    Build the training and evaluation dataloaders from plain config values.

    Returns
    -------
    train_dataloader :
        RandomSampleDataset over the train split, with an InfiniteSampler - random windows,
        drawn endlessly. Random windows are what you want for TRAINING: they decorrelate
        episodes and expose the policy to the whole split.
    eval_dataloader :
        BuildingTemperatureDataset over the test split - consecutive, deterministic windows,
        walked once. Determinism is the point: this is the metric PBT selects trials on, and
        scoring the same trial twice must give the same number. A RandomSampleDataset here
        would make the metric jitter with the draw, and PBT would exploit sampling luck.

    data_config keys:
        "csv_path"    : path to the room temperature .csv
        "sample_size" : rows of data per episode
        "warmup_size" : rows reserved at the start for warming up the latent nodes (default 0)
        "dt"          : timestep the data is resampled to, seconds (default 30)

    NOTE: the two dataset classes compute their test split the same way only when
    warmup_size is 0 (RandomSampleDataset subtracts the warmup twice - see its
    _split_dataset). With a non-zero warmup the evaluation windows would not line up with
    the split the training loader avoids, so that combination is rejected here rather than
    silently leaking training data into the metric.
    """
    csv_path = data_config["csv_path"]
    sample_size = data_config["sample_size"]
    warmup_size = data_config.get("warmup_size", 0)
    dt = data_config.get("dt", 30)

    if warmup_size:
        raise NotImplementedError(
            "make_dataloaders only supports warmup_size=0 - RandomSampleDataset._split_dataset "
            "subtracts the warmup from the test split's offset as well as from the total, so the "
            "train and evaluation splits would overlap. Fix that split before using a warmup here."
        )

    path_sorted = sort_data(str(csv_path), dt)
    with FileLock(f"{os.path.dirname(os.path.abspath(path_sorted))}.lock"):
        train_dataset = RandomSampleDataset(
            path_sorted, sample_size, warmup_size, train=True, test=False, epoch_length=data_config.get("epoch_length")
        )
        eval_dataset = BuildingTemperatureDataset(path_sorted, sample_size, all=False, train=False, test=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        sampler=InfiniteSampler(train_dataset),
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    return train_dataloader, eval_dataloader


def change_origin(room_coordinates):
    """Shifts a list of room coordinate polygons so [0,0] sits at the
    minimum x/y corner across all rooms combined, preserving each room's
    position relative to the others."""
    all_points = [point for room in room_coordinates for point in room]
    x0 = min(point[0] for point in all_points)
    y0 = min(point[1] for point in all_points)

    shifted_rooms = []
    for room in room_coordinates:
        shifted_rooms.append([[round((x - x0) / 10, 2), round((y - y0) / 10, 2)] for x, y in room])
    return shifted_rooms


def initialise_model(
    cooling_policy, scaling, weather_data_outdoor_temperature, weather_data_UTC_time, room_names, room_coordinates
):
    room_coordinates = change_origin(room_coordinates)

    rooms = []
    for i in range(len(room_names)):
        rooms.append(Room(room_names[i], room_coordinates[i]))

    # Initialise Building
    bld = Building(rooms)

    Tout = torch.tensor(weather_data_outdoor_temperature)
    t = torch.tensor(weather_data_UTC_time)
    Tout_continuous = Interp1D(t, Tout, method="linear")  # Interp1D object

    # Initialise RCModel with the building.
    # transform=None: parameters are held directly in 0-1 machine space. The sigmoid
    # parameterisation existed to keep gradient descent on unbounded parameters well
    # behaved; the search supplies bounded parameters directly, so it would only obscure
    # what a perturbation actually does.
    model = RCModel(bld, scaling, Tout_continuous, transform=None, cooling_policy=cooling_policy)

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

        path_sorted = path if path[-11:] == "_sorted.csv" else path[:-4] + "_sorted.csv"

        # Sort df by time (raw data not always in order)
        df = df.sort_values(by=["time"], ascending=True)

        # insert date-time value at start of df
        try:
            df.insert(loc=0, column="date-time", value=pd.to_datetime(df["time"], unit="ms"))
        except ValueError as err:
            raise ValueError(
                "Data appears to have already been sorted. Check if still appropriate and add _sorted.csv tag to avoid"
                "this error."
            ) from err

        # downscale data to a frequency of dt (seconds) use the mean value and round to 2dp.
        df = df.set_index("date-time").resample(str(dt) + "s").mean().round(2)

        # time column is converted to unix epoch seconds to match the date-time
        df["time"] = (df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        # change date-time from UTC to Local time
        infer_dst = np.array(
            [False] * df.shape[0]
        )  # all False -> every row considered DT, alternative is True to indicate DST.
        #    The array must correspond to the iloc of df.index
        df = df.tz_localize("Europe/London", ambiguous=infer_dst, nonexistent="shift_forward")  # causes error so commented out

        df = df.interpolate().round(2)  # interpolate missing values NaN

        df.to_csv(path_sorted, index=True)

    def need_to_sort(path, dt):

        def get_dt(path):
            df_dt = pd.read_csv(path)["time"][0:2].values
            return df_dt[1] - df_dt[0]

        # Does path already have sorted tag?
        if path[-11:] == "_sorted.csv":
            # if so, is dt correct? if not, re-sort
            return get_dt(path) != dt

        # path does not contain _sorted.csv
        else:
            # Does path_sorted exist?
            path_sorted = path[:-4] + "_sorted.csv"
            import os.path

            if os.path.isfile(path_sorted):  # check if file already exists
                # if file exists check if dt is correct; if not, re-sort
                return get_dt(path_sorted) != dt

            else:  # File doesn't exist
                return True

    if need_to_sort(path, dt):
        sort(path, dt)

    # return the path_sorted
    path_sorted = path if path[-11:] == "_sorted.csv" else path[:-4] + "_sorted.csv"

    return path_sorted


def model_to_csv(observations, output_path):
    """
    Produces a .csv of the output. To then be used in return_to_sender.py
    """
    # Produce a .csv in the same format as current data, retains compatibility with dataloader
    titles = [["date-time", "time"], [f"Rm{i}" for i in range(observations[:, 3:].shape[1])]]
    titles = [item for sublist in titles for item in sublist]

    # sort into date-time, unix time and temp data columns. (date-time exists just to keep format consistent)
    df = torch.hstack((torch.zeros(len(observations), 1) * torch.nan, observations[:, 0].unsqueeze(1), observations[:, 3:]))
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
        raise TypeError("n must be even.")

    y_a = y[-n : -n // 2]
    y_b = y[-n // 2 :]

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
                raise ValueError("Trying to smooth entire array, don't include y_hat")

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
                        for yi in y[i + 1 :]:
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
    t0 = 4 * 24 * 60**2  # buffer to go from thursday to monday
    time = torch.linspace(0 + t0, 24 * 60**2 + t0, n)
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
                log_prob = info["action_logp"]
                # Get prob of getting 1:
                if action == 1:
                    pr = torch.e**log_prob  # Convert log_prob to normal prob.
                elif action == 0:
                    pr = 1 - torch.e**log_prob  # pr(a=1) = 1 - pr(a=0)
                else:
                    raise ValueError(f"action={action}, must be exactly 1 or 0.")

                img[i, j] = pr

    fig = plt.figure()
    plt.imshow(img, origin="lower", aspect="auto", cmap="viridis", extent=(0, 24, bounds[0], bounds[1]), vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel("Time of Day [hours]")
    plt.ylabel(r"Indoor Temperature [$^\circ$C]")
    plt.title("Policy Plot")
    plt.xticks(np.linspace(0, 24, 13))
    plt.yticks(np.linspace(bounds[0], bounds[1], 7))
    plt.grid(
        color="k",
        linestyle="--",
    )

    if path:
        fig.savefig(path)
    else:
        plt.show()
