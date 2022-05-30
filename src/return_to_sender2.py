import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange
from xitorch.interpolate import Interp1D
from torch.utils.data import TensorDataset

from rcmodel import *
from rcmodel.tools.helper_functions import model_to_csv


def env_creator(env_config, preprocess=True):
    model_config = env_config["model_config"]
    model = model_creator(model_config)
    time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
    model.iv_array = model.get_iv_array(time_data)

    env_config["RC_model"] = model

    env = LSIEnv(env_config)

    # wrap environment:
    if preprocess:
        env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

    return env


# ----- Create Original model and produce a dummy data csv file: -----

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
dummy_data_path = './outputs/return_to_sender_dummy_data.csv'

from pathlib import Path
Path('./outputs').mkdir(parents=True, exist_ok=True)

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'

# load a model which we will then try to replicate.
load_policy = './prior_policy.pt'
load_physical = './return_model.pt'

# original model config dict:
model_og_config = {
    "rm_cap_per_area_min": 1e2,
    "rm_cap_per_area_max": 1e4,
    "external_capacitance_min": 1e3,
    "external_capacitance_max": 1e8,
    "resistance_min": 0.1,
    "resistance_max": 5,
    "Q_max": 300,
    "room_names": ["seminar_rm_a_t0106"],
    "room_coordinates": [[[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74],
                          [172.64, 231.74], [172.64, 125.94]]],
    "weather_data_path": weather_data_path,
    "room_data_path": dummy_data_path,  # This is just so the environment will work and is used to create the reward
    "load_model_path_policy": load_policy,  # or None
    "load_model_path_physical": load_physical,  # or None
}

num_days = 2
t_eval = torch.arange(0, num_days * 24 * 60 ** 2, 30,
                      dtype=torch.float64) + 1622505600  # + min start time of weather data
# create a csv file of the timeseries we want.
pd.DataFrame([t_eval.numpy(), np.ones(len(t_eval))*25]).T.to_csv(dummy_data_path)


sample_size = len(t_eval)
train_dataset_og = BuildingTemperatureDataset(dummy_data_path, sample_size, all=True)

env_config_og = {"model_config": model_og_config,
                 "dataloader": torch.utils.data.DataLoader(train_dataset_og, batch_size=1, shuffle=False),
                 "step_length": 15  # minutes passed in each step.
                 }

env_og = env_creator(env_config_og, preprocess=True)

observations = []
for i in range(len(train_dataset_og)):
    done = False
    observation = env_og.reset()
    while not done:
        env_og.render()
        action = env_og.RC.cooling_policy.get_action(torch.tensor(observation, dtype=torch.float32)).item()
        observation, reward, done, _ = env_og.step(action)
        observations.append(env_og.env.observation)

observations = torch.concat(observations)


output_path = './outputs/return_to_sender_out_sorted.csv'
model_to_csv(observations, output_path)

iv_array = Interp1D(t_eval, observations[:, 1:].T.detach(), method='linear')

# see what the original model looks like:
# pltsolution_1rm(model_origin, prediction=pred, time=t_eval)


# with open('original_params.txt', 'w') as f:
#     params = model_origin.scaling.physical_param_scaling(model_origin.transform(model_origin.params)).detach().numpy()
#     loads = model_origin.scaling.physical_cooling_scaling(model_origin.transform(model_origin.loads))
#     cooling = loads[0, :].detach().numpy()
#     gain = loads[1, :].detach().numpy()
#
#     all_params = np.concatenate((params, gain, cooling))
#     params_heading = ['Rm Cap/m2 (J/K.m2)', 'Ext Wl Cap 1 (J/K)', 'Ext Wl Cap 2 (J/K)', 'Ext Wl Res 1 (K.m2/W)',
#                       'Ext Wl Res 2 (K.m2/W)', 'Ext Wl Res 3 (K.m2/W)', 'Int Wl Res (K.m2/W)', 'Offset Gain (W/m2)',
#                       'Cooling (W)']
#
#     for i in range(len(all_params)):
#         f.write(f'{params_heading[i]}: {all_params[i]:.3f} \n')

# -----------------------------------------------------

# Run current optimisation process and see if we can find the original parameters:


model_config = {
    "rm_cap_per_area_min": 1e2,
    "rm_cap_per_area_max": 1e4,
    "external_capacitance_min": 1e3,
    "external_capacitance_max": 1e8,
    "resistance_min": 0.1,
    "resistance_max": 5,
    "Q_max": 300,
    "room_names": ["seminar_rm_a_t0106"],
    "room_coordinates": [[[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74],
                          [172.64, 231.74], [172.64, 125.94]]],
    "weather_data_path": '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv',
    "room_data_path": output_path,
    "load_model_path_policy": load_policy,  # or None
    "load_model_path_physical": load_physical,  # or None
}

# # Initialise Optimise Class - for training physical model
dt = 30
sample_size = 24 * 60 ** 2 / dt  # ONE DAY
# warmup_size = sample_size * 7  # ONE WEEK
warmup_size = 0
train_dataset = BuildingTemperatureDataset(model_config['room_data_path'], sample_size, all=True)
# test_dataset = RandomSampleDataset(model_config['room_data_path'], sample_size, warmup_size, train=False, test=True)

config = {
    "env": LSIEnv,  # or "corridor" if registered above
    "env_config": {"model_config": model_config,
                   "dataloader": torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False),
                   "step_length": 15  # minutes passed in each step.
                   },

    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    "num_workers": 8,  # parallelism
    "framework": "torch",
}

env = env_creator(config["env_config"])
env.RC.iv_array = iv_array  # correct output from original model so our iv is always correct

for i in range(len(train_dataset)):
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = env.RC.cooling_policy.get_action(torch.tensor(observation, dtype=torch.float32)).item()
        observation, reward, done, _ = env.step(action)
