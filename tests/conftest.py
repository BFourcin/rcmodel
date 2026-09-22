import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from rcmodel import Building, BuildingTemperatureDataset, InputScaling, RandomSampleDataset, RCModel, Room

T_ORIGIN = 1_600_000_000  # realistic unix-epoch-scale start time (2020-09-13), deliberately not 0.
# get_iv_array() once had a bug where it queried an Interp1D built over absolute (epoch-scale)
# time using a relative (starts-at-zero) time array. Every fixture below started at t=0, which
# made that shift a silent no-op and completely hid the bug - hence the non-zero origin here.


@pytest.fixture
def rooms_n2():
    """
    Two rooms side by side
    """
    rm1 = Room("rm1", [[0, 0], [5, 0], [5, 5], [0, 5]])
    rm2 = Room("rm2", [[5, 0], [10, 0], [10, 5], [5, 5]])
    rooms = [rm1, rm2]

    return rooms


@pytest.fixture
def rooms_n9():
    """
    Complicated arrangement of rooms
    """
    rooms = []

    rooms.append(Room("rm1", [[0, 0], [5, 0], [5, 5], [0, 5]]))
    rooms.append(Room("rm2", [[5, 0], [10, 0], [10, 2], [10, 4], [10, 5], [5, 5]]))
    rooms.append(Room("rm3", [[0, 5], [5, 5], [10, 5], [10, 6], [10, 8], [10, 10]]))

    rooms.append(Room("rm4", [[10, 10], [12, 10], [12, 8], [10, 8]]))
    rooms.append(Room("rm5", [[10, 8], [12, 8], [12, 6], [10, 6]]))
    rooms.append(Room("rm6", [[10, 6], [12, 6], [12, 4], [10, 4], [10, 5]]))
    rooms.append(Room("rm7", [[10, 4], [12, 4], [12, 2], [10, 2]]))
    rooms.append(Room("rm8", [[10, 2], [12, 2], [12, 0], [10, 0]]))
    rooms.append(
        Room(
            "rm9",
            [[12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [14, 10], [14, 0]],
        )
    )

    return rooms


def get_building(rooms):
    height = 1
    rm_cap = [300]
    Ce = [1e3, 8e2]
    Re = [5, 1, 0.5]
    Rint = [0.1]

    theta = [rm_cap, Ce, Re, Rint]
    theta = [item for sublist in theta for item in sublist]  # flatten list

    bld = Building(rooms, height)
    bld.update_inputs(theta)

    return bld


@pytest.fixture
def building_n2(rooms_n2):
    return get_building(rooms_n2)


@pytest.fixture
def building_n9(rooms_n9):
    return get_building(rooms_n9)


def get_model(building):
    # Function for constant outside temperature. try/except allows for broadcasting of value.
    def dummy_tout(t):
        try:
            return -5 * torch.ones(len(t))

        except TypeError:
            return torch.tensor(-5)

    rm_CA = [200, 800]  # [min, max] Capacitance/m2
    C1 = [1.5 * 10**4, 10**6]  # Capacitance
    C2 = [1.5 * 10**4, 10**6]
    R1 = [0.2, 1.2]  # Resistance ((K.m^2)/W)
    R2 = [0.2, 1.2]
    R3 = [0.2, 1.2]
    Rin = [0.2, 1.2]
    cool = [0, 5000]  # Cooling limit in W/m2
    gain = [0, 5000]  # gain limit (W/m^2) no additional energy

    scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain)

    # Initialise RCModel with the building and InputScaling.
    # transform=None: parameters are held directly in 0-1 machine space - see RCModel.
    model = RCModel(building, scaling, dummy_tout, transform=None)
    return model


@pytest.fixture
def model_n2(building_n2):
    return get_model(building_n2)


@pytest.fixture
def model_n9(building_n9):
    return get_model(building_n9)


@pytest.fixture
def fake_time():
    """Shared time axis (s) for synthetic weather + indoor-temperature data, so both
    always cover the exact same domain. Sized to match the 6-hour/720-row shape the
    old real data file had, keeping batch counts for the optimiser tests similar.
    Starts at T_ORIGIN (realistic epoch scale), not 0 - see the module-level comment."""
    dt = 30
    n_seconds = 6 * 60**2
    return T_ORIGIN + np.arange(0, n_seconds, dt)


@pytest.fixture
def fake_outdoor_weather(fake_time):
    """Synthetic outdoor temperature series over `fake_time`."""
    t_rel = fake_time - fake_time[0]
    n_seconds = t_rel[-1] + (t_rel[1] - t_rel[0])
    return 10 + 5 * np.sin(2 * np.pi * t_rel / n_seconds)


@pytest.fixture
def fake_rooms():
    fake_room_coordinates = [
        [[0, 0], [5, 0], [5, 5], [0, 5]],
        [[5, 0], [10, 0], [10, 2], [10, 4], [10, 5], [5, 5]],
        [[0, 5], [5, 5], [10, 5], [10, 6], [10, 8], [10, 10]],
        [[10, 10], [12, 10], [12, 8], [10, 8]],
        [[10, 8], [12, 8], [12, 6], [10, 6]],
        [[10, 6], [12, 6], [12, 4], [10, 4], [10, 5]],
        [[10, 4], [12, 4], [12, 2], [10, 2]],
        [[10, 2], [12, 2], [12, 0], [10, 0]],
        [[12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [14, 10], [14, 0]],
    ]
    fake_room_names = ["rm1", "rm2", "rm3", "rm4", "rm5", "rm6", "rm7", "rm8", "rm9"]
    return fake_room_names, fake_room_coordinates


@pytest.fixture
def synthetic_indoor_temperature_csv(tmp_path, fake_time, fake_rooms):
    """Writes a synthetic per-room indoor-temperature CSV over the same time domain as
    `fake_outdoor_weather`, in the (date-time, time, <room columns>) format
    RandomSampleDataset/BuildingTemperatureDataset expect."""
    rng = np.random.default_rng(7)
    t_rel = fake_time - fake_time[0]
    n_seconds = t_rel[-1] + (t_rel[1] - t_rel[0])
    base_indoor = 22 + 2 * np.sin(2 * np.pi * (t_rel - 3 * 60**2) / n_seconds)
    fake_room_names, _ = fake_rooms
    room_offsets = rng.uniform(-1.0, 1.0, size=len(fake_room_names))
    indoor_temps = base_indoor[:, None] + room_offsets[None, :]

    df = pd.DataFrame(indoor_temps, columns=fake_room_names)
    df.insert(0, "time", fake_time)
    df.insert(0, "date-time", pd.to_datetime(fake_time, unit="s"))

    csv_path = tmp_path / "synthetic_indoor_temperature.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def get_model_config(fake_time, fake_outdoor_weather, fake_rooms):
    np.random.seed(42)
    fake_room_names, fake_room_coordinates = fake_rooms

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
        "room_names": fake_room_names,
        "room_coordinates": fake_room_coordinates,
        "weather_data_outdoor_temperature": fake_outdoor_weather,
        "weather_data_UTC_time": fake_time,
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
        },
    }
    return model_config


@pytest.fixture
def get_datasets(synthetic_indoor_temperature_csv):
    csv_path = synthetic_indoor_temperature_csv
    dt = 30  # seconds
    sample_size = 1 * 60**2 / dt  # ONE HOUR
    warmup_size = 0
    train_dataset = RandomSampleDataset(csv_path, sample_size, warmup_size, train=True, test=False)
    test_dataset = RandomSampleDataset(csv_path, sample_size, warmup_size, train=False, test=True)
    return train_dataset, test_dataset


@pytest.fixture
def full_building_dataset(synthetic_indoor_temperature_csv, fake_time):
    """The whole of `synthetic_indoor_temperature_csv`, deterministically (no random
    windowing) - for tests that need the full dataset a model would see, e.g. get_iv_array()."""
    return BuildingTemperatureDataset(synthetic_indoor_temperature_csv, sample_size=len(fake_time), all=True)


@pytest.fixture
def sorted_csv(synthetic_indoor_temperature_csv, tmp_path):
    """The synthetic data under a `*_sorted.csv` name.

    sort_data() treats an unsuffixed path as raw data and re-sorts it, and raw data is
    expected to carry MILLISECOND timestamps (it does pd.to_datetime(..., unit="ms")). The
    fixtures write seconds, matching what the dataset classes read back, so re-sorting would
    reinterpret every timestamp as being in 1970. The suffix tells sort_data the file is
    already in its final form.
    """
    destination = tmp_path / "synthetic_indoor_temperature_sorted.csv"
    shutil.copy(synthetic_indoor_temperature_csv, destination)
    return destination


@pytest.fixture
def data_config(sorted_csv):
    """Plain-data description of the dataset, as make_dataloaders() expects."""
    dt = 30
    return {
        "csv_path": str(sorted_csv),
        "sample_size": int(1 * 60**2 / dt),  # ONE HOUR
        "warmup_size": 0,
        "dt": dt,
    }


@pytest.fixture
def env_config(data_config):
    """env_creator config holding only plain data - no live RCModel.

    This is the shape a PBT trial uses: everything the environment needs is serialisable,
    so a trial's parameters are whatever its config says they are.
    """
    return {
        "data_config": data_config,
        "step_length": 15,  # minutes passed in each step.
        "render_mode": None,
        "model_config": None,  # filled in per test from get_model_config
    }


@pytest.fixture
def physical_params(get_model_config):
    """Two distinct, plausible physical parameter sets, at 25% and 75% of every range.

    Used to check that a change of parameters actually reaches the model - two sets that
    differ in every dimension make an accidental pass very unlikely.
    """

    def at_fraction(fraction):
        values = {}
        for key in ("C_rm", "C1", "C2", "R1", "R2", "R3", "Rin", "cool", "gain"):
            low, high = get_model_config[key]
            values[key] = low + fraction * (high - low)
        return values

    return at_fraction(0.25), at_fraction(0.75)
