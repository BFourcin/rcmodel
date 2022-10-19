import gym
import ray
import torch
import pandas as pd
import numpy as np
import time
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray import tune
from tqdm import trange

from rcmodel import *

start_time = time.time()

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
# csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/summer2021_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

torch.set_num_threads(1)

dt = 30
sample_size = 24 * 60 ** 2 / dt  # ONE DAY
# warmup_size = sample_size * 7  # ONE WEEK
warmup_size = 0

train_dataloader, test_dataloader = dataloader_creator(csv_path, sample_size, warmup_size, dt=30)

# dataloader to use:
dataloader = train_dataloader

n_workers = 7  # workers per env instance

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
    "load_model_path_policy": None,  # './prior_policy.pt',  # or None
    "load_model_path_physical": 'trained_model.pt',  # or None
    "parameters": {
        "C_rm": tune.uniform(0, 1),
        "C1": tune.uniform(0, 1),
        "C2": tune.uniform(0, 1),
        "R1": tune.uniform(0, 1),
        "R2": tune.uniform(0, 1),
        "R3": tune.uniform(0, 1),
        "Rin": tune.uniform(0, 1),
        "cool": tune.uniform(0, 1),  # 0.09133423646610082
        "gain": tune.uniform(0, 1),  # 0.9086668150306394
    }
    # "parameters": {
    #         "C_rm": 0.7,
    #         "C1": 0.7,
    #         "C2": 0.7,
    #         "R1": 0.7,
    #         "R2": 0.7,
    #         "R3": 0.7,
    #         "Rin": 0.7,
    #         "cool": 0.7,  # 0.09133423646610082
    #         "gain": 0.7,  # 0.9086668150306394
    #     }
}

# Initialise Model
# model = model_creator(model_config)

# dt = 30
# sample_size = 24 * 60 ** 2 / dt  # ONE DAY
# # warmup_size = sample_size * 7  # ONE WEEK
# warmup_size = 7 * sample_size
#
# train_dataloader, test_dataloader = dataloader_creator(model_config['room_data_path'], sample_size, warmup_size, dt=30)
# train_dataset = RandomSampleDataset(model_config['room_data_path'], sample_size, warmup_size, train=True, test=False)
# test_dataset = RandomSampleDataset(model_config['room_data_path'], sample_size, warmup_size, train=False, test=True)


# Get iv array because we can pre calc for this experiment
def func_iv_array(model_config, dataset):
    mc = model_config.copy()
    mc["cooling_param"], mc["gain_param"] = None, None
    model = model_creator(mc)
    return get_iv_array(model, dataset)


ppo_config = {
    "env": "LSIEnv",
    "env_config": {"model_config": model_config,
                   "dataloader": dataloader,
                   "step_length": 15,  # minutes passed in each step.
                   # "iv_array": func_iv_array(model_config, dataloader.dataset),  # because phys parameters aren't changing.
                   "iv_array": None,
                   "render_mode": 'single_rgb_array',  # "single_rgb_array"
                   },

    # PPO Stuff:
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    # "lr": 0.01,
    "num_workers": n_workers,  # workers per env instance
    "framework": "torch",
    "vf_clip_param": 3000,
    "rollout_fragment_length": 96 * 5,  # number of steps per rollout
    "train_batch_size": 96 * 5 * n_workers,
    "sgd_minibatch_size": 96,
    "horizon": None,
    "render_env": False,
    # "monitor": True
}


def env_creator(env_config):
    with torch.no_grad():
        model_config = env_config["model_config"]
        model = model_creator(model_config)
        # model.iv_array = model.get_iv_array(time_data)

        if not env_config["iv_array"]:
            model.iv_array = get_iv_array(model, env_config["dataloader"].dataset)
        else:
            model.iv_array = env_config["iv_array"]

        env_config["RC_model"] = model

        env = LSIEnv(env_config)

        # wrap environment:
        env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

    return env

# env = env_creator(ppo_config["env_config"])
# for i in range(len(dataloader.dataset)):
#     done = False
#     observation = env.reset()
#     while not done:
#         env.render()
#         # action = np.random.randint(2)
#         action = env.RC.cooling_policy.get_action(torch.tensor(observation, dtype=torch.float32)).item()
#         observation, reward, done, _ = env.step(action)


register_env("LSIEnv", env_creator)

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch

hebo = HEBOSearch(metric="episode_reward_mean", mode="max",)
hebo = tune.suggest.ConcurrencyLimiter(hebo, max_concurrent=8)

# AsyncHyperBand enables aggressive early stopping of bad trials.
scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    grace_period=2,
    max_t=2000,
)


print(f'Completed setup & calculated IV array in: {time.time()- start_time:.1f} seconds')

ray.init(num_cpus=8)

# Will fail if env initialised before tune()
tune.run(
    "PPO",
    stop={"episode_reward_mean": -2,
          "training_iteration": 2000},
    config=ppo_config,
    num_samples=400,  # number of trials to perform
    scheduler=scheduler,
    search_alg=hebo,
    local_dir="./outputs/checkpoints/tuned_trial",
    checkpoint_freq=5,
    checkpoint_at_end=True,
    checkpoint_score_attr="episode_reward_mean",
    # keep_checkpoints_num=5,
    # reuse_actors=True
    sync_config=tune.SyncConfig(),
    verbose=3,
    # Verbosity mode. 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results. Defaults to 3.
    # fail_fast="raise",
    # resume="AUTO",  # resume from the last run specified in sync_config
)

ray.shutdown()
