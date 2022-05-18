import gym
import ray
import torch
import pandas as pd
import time
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray import tune
from tqdm import trange

from rcmodel import *

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data


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
    "room_data_path": '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv',
    "load_model_path_policy": './prior_policy.pt',  # or None
    "load_model_path_physical": './prior_physical.pt',  # or None
}

# Initialise Model
# model = model_creator(model_config)


# # Initialise Optimise Class - for training physical model
dt = 30
sample_size = 24 * 60 ** 2 / dt  # ONE DAY
# warmup_size = sample_size * 7  # ONE WEEK
warmup_size = 0
train_dataset = RandomSampleDataset(model_config['room_data_path'], sample_size, warmup_size, train=True, test=False)
test_dataset = RandomSampleDataset(model_config['room_data_path'], sample_size, warmup_size, train=False, test=True)

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


def env_creator(env_config):
    model_config = env_config["model_config"]
    model = model_creator(model_config)
    time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
    model.iv_array = model.get_iv_array(time_data)

    print("env-creator")

    env_config["RC_model"] = model

    env = LSIEnv(env_config)
    # wrap environment:
    env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

    return env

# import numpy as np
# env = env_creator(config["env_config"])

# for i in range(len(train_dataset)):
#     done = False
#     env.reset()
#     while not done:
#         env.render()
#         action = np.random.randint(2)
#         observation, reward, done, _ = env.step(action)


register_env("LSIEnv", env_creator)

# ray.init()
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config["vf_clip_param"] = 3000
ppo_config.update(config)

trainer = ppo.PPOTrainer(env="LSIEnv", config=ppo_config)

# Can optionally call trainer.restore(path) to load a checkpoint.

keys = ["episode_reward_min", "episode_reward_max", "episode_reward_mean", "episodes_this_iter", "episodes_total"]

for i in trange(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()

    string = ""
    for key in keys:
        string += key + f": {result[key]:.2f}, "

    print(string)

    # print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)



# tune.run(
#     "PPO",
#     # stop={"episode_reward_mean": 200},
#     config=ppo_config,
#     checkpoint_at_end=True,
#     reuse_actors=True
# )
