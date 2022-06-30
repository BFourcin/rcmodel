import gym
import torch
import pandas as pd
import numpy as np
import time
from ray import tune
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt

from rcmodel import *


def env_creator(env_config):
    with torch.no_grad():
        model_config = env_config["model_config"]
        model = model_creator(model_config)
        time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
        model.iv_array = get_ivarr()
        # model.iv_array = model.get_iv_array(time_data)

        env_config["RC_model"] = model

        env = LSIEnv(env_config)

        # wrap environment:
        env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

    return env


def get_ivarr():
    from xitorch.interpolate import Interp1D
    def env_creator2(env_config, preprocess=True):
        model_config = env_config["model_config"]
        model = model_creator(model_config)
        time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1],
                                 dtype=torch.float64)
        model.iv_array = model.get_iv_array(time_data)

        env_config["RC_model"] = model

        env = LSIEnv(env_config)

        # wrap environment:
        if preprocess:
            env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

        return env
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
        "C_rm": [1e2, 1e4],  # [min, max] Capacitance/m2
        "C1": [1e3, 1e8],  # Capacitance
        "C2": [1e3, 1e8],
        "R1": [0.1, 5],  # Resistance ((K.m^2)/W)
        "R2": [0.1, 5],
        "R3": [0.1, 5],
        "Rin": [0.1, 5],
        "cool": [0, 300],  # Cooling limit in W/m2
        "gain": [0, 300],  # Gain limit in W/m2
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
    pd.DataFrame([t_eval.numpy(), np.ones(len(t_eval)) * 25]).T.to_csv(dummy_data_path)

    sample_size = len(t_eval)
    train_dataset_og = BuildingTemperatureDataset(dummy_data_path, sample_size, all=True)

    env_config_og = {"model_config": model_og_config,
                     "dataloader": torch.utils.data.DataLoader(train_dataset_og, batch_size=1, shuffle=False),
                     "step_length": 15  # minutes passed in each step.
                     }

    env_og = env_creator2(env_config_og)

    observations = []
    for i in range(len(train_dataset_og)):
        done = False
        observation = env_og.reset()
        while not done:
            # env_og.render()
            action = env_og.RC.cooling_policy.get_action(torch.tensor(observation, dtype=torch.float32)).item()
            observation, reward, done, _ = env_og.step(action)
            observations.append(env_og.env.observation)

    observations = torch.concat(observations)

    output_path = './outputs/return_to_sender_out_sorted.csv'
    model_to_csv(observations, output_path)

    iv_array = Interp1D(t_eval, observations[:, 1:].T.detach(), method='linear')

    del observations
    return iv_array

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
csv_path = './outputs/return_to_sender_out_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

torch.set_num_threads(1)

n_workers = 3  # workers per env instance

model_config = {
    # Ranges:
    "C_rm": [1e2, 1e4],  # [min, max] Capacitance/m2
    "C1": [1e3, 1e8],  # Capacitance
    "C2": [1e3, 1e8],
    "R1": [0.1, 5],  # Resistance ((K.m^2)/W)
    "R2": [0.1, 5],
    "R3": [0.1, 5],
    "Rin": [0.1, 5],
    "cool": [0, 300],  # Cooling limit in W/m2
    "gain": [0, 300],  # Gain limit in W/m2
    "room_names": ["seminar_rm_a_t0106"],
    "room_coordinates": [[[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74],
                          [172.64, 231.74], [172.64, 125.94]]],
    "weather_data_path": weather_data_path,
    "room_data_path": csv_path,
    "load_model_path_policy": './prior_policy.pt',  # or None
    "load_model_path_physical": 'trained_model.pt',  # or None
    "cooling_param": None,
    "gain_param": None,
}

# Initialise Model
# model = model_creator(model_config)

# inputs = torch.tensor(
#     [0.908598186, 0.02005488, 0.19026246, 0.036782882, 0.005074634, 0.896428514, 0.869218101, 0.494917939, 0.001629876])
# in_params = torch.logit(inputs[0:7])
# in_loads = torch.logit(torch.reshape(inputs[7:], model.loads.shape))
#
# model.params = torch.nn.Parameter(in_params)
# model.loads = torch.nn.Parameter(in_loads)
#
# time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
# model.iv_array = model.get_iv_array(time_data)

# # Initialise Optimise Class - for training physical model
dt = 30
sample_size = 5 * 24 * 60 ** 2 / dt
# warmup_size = sample_size * 7  # ONE WEEK
warmup_size = 0

plot_results_data = BuildingTemperatureDataset(model_config['room_data_path'], sample_size, all=True)
plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

Path("./outputs/plots/results/").mkdir(parents=True, exist_ok=True)

# pltsolution_1rm(model, plot_dataloader, './outputs/plots/results/NmmsoResult.png')

env_config = {"model_config": model_config,
              "dataloader": plot_dataloader,
              "step_length": 15  # minutes passed in each step.
              }

env = env_creator(env_config)

class MyProblem:
    def __init__(self, env, dataset):
        self.env = env
        self.dataset = dataset

        self.env.RC.transform = None  # turn sigmoid off

    def fitness(self, params):
        with torch.no_grad():
            n = self.env.RC.building.n_params
            self.env.RC.params = torch.nn.Parameter(torch.tensor(params[0:n], dtype=torch.float32))
            loads = torch.tensor(np.array([params[n:n + 2 * len(self.env.RC.building.rooms)]]), dtype=torch.float32)
            self.env.RC.loads = torch.nn.Parameter(loads.reshape(2, len(self.env.RC.building.rooms)))

            total_reward = 0
            for i in range(len(self.dataset)):
                done = False
                observation = self.env.reset()
                while not done:
                    fig = self.env.render()
                    action = self.env.RC.cooling_policy.get_action(
                        torch.tensor(observation, dtype=torch.float32)).item()
                    observation, reward, done, _ = self.env.step(action)
                    total_reward += reward

                print(total_reward)

            return fig  # pynmmso maximises

    def get_bounds(self):
        n = self.env.RC.building.n_params + 2 * len(self.env.RC.building.rooms)
        return list(np.zeros(n)), list(np.ones(n))


p = MyProblem(env, plot_results_data)

input_params = [0.999368892,0.017527581,0.099818035,0.941952717,0.101520705,0.833274993,0.185717194,0.813527961,0.001403869]
fig = p.fitness(input_params)


fig.savefig('./outputs/plots/results/NmmsoResult.png')
