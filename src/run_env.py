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

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

torch.set_num_threads(1)

n_workers = 1  # workers per env instance

# "C_rm": [1e2, 1e4],  # [min, max] Capacitance/m2
# "C1": [1e3, 1e8],  # Capacitance
# "C2": [1e3, 1e8],
# "R1": [0.1, 5],  # Resistance ((K.m^2)/W)
# "R2": [0.1, 5],
# "R3": [0.1, 5],
# "Rin": [0.1, 5],
# "cool": [0, 300],  # Cooling limit in W/m2
# "gain": [0, 300],  # Gain limit in W/m2

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
    "room_data_path": csv_path,
    "load_model_path_policy": None,  # './prior_policy.pt',  # or None
    "load_model_path_physical": 'trained_model.pt',  # or None
    "cooling_param": tune.uniform(0, 1),
    "gain_param": tune.uniform(0, 1),
    # "cooling_param": 0.5,
    # "gain_param": 0.5,
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


# Get iv array because we can pre calc for this experiment
def func_iv_array(model_config):
    mc = model_config.copy()
    mc["cooling_param"], mc["gain_param"] = None, None
    model = model_creator(mc)
    time_data = torch.tensor(pd.read_csv(mc['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
    return model.get_iv_array(time_data)


ppo_config = {
    "env": "LSIEnv",  # or "corridor" if registered above
    "env_config": {"model_config": model_config,
                   "dataloader": torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False),
                   "step_length": 15,  # minutes passed in each step.
                   "iv_array": func_iv_array(model_config),  # because phys parameters aren't changing.
                   # "cooling_param": tune.uniform(0, 1),
                   # "gain_param": tune.uniform(0, 1)
                   },

    # PPO Stuff:
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    # "lr": 0.01,
    "num_workers": n_workers,  # parallelism
    "framework": "torch",
    "vf_clip_param": 3000,
    "rollout_fragment_length": 96 * 5,  # number of steps per rollout
    "train_batch_size": 96 * 5 * n_workers,
    "sgd_minibatch_size": 96,
    "horizon": None,
}


def env_creator(env_config):
    with torch.no_grad():
        # global config
        # print(config)
        model_config = env_config["model_config"]
        # print(tune_parameters)
        # model_config["cooling_param"] = env_config["cooling_param"]
        # model_config["gain_param"] = env_config["gain_param"]
        print(model_config["cooling_param"], model_config["gain_param"])
        model = model_creator(model_config)
        # time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
        # model.iv_array = model.get_iv_array(time_data)
        model.iv_array = env_config["iv_array"]

        env_config["RC_model"] = model

        env = LSIEnv(env_config)

        # wrap environment:
        env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

    return env


# env = env_creator(ppo_config["env_config"])
# for i in range(len(train_dataset)):
#     done = False
#     observation = env.reset()
#     while not done:
#         # env.render()
#         # action = np.random.randint(2)
#         action = env.RC.cooling_policy.get_action(torch.tensor(observation, dtype=torch.float32)).item()
#         observation, reward, done, _ = env.step(action)


register_env("LSIEnv", env_creator)

# ray.init()
# ppo_config2 = ppo.DEFAULT_CONFIG.copy()
# ppo_config["vf_clip_param"] = 3000
# ppo_config["rollout_fragment_length"] = 96 * 5  # number of steps per rollout
# ppo_config["train_batch_size"] = 96 * 5 * n_workers
# ppo_config["sgd_minibatch_size"] = 96
# ppo_config["horizon"] = None  # Number of steps after which the episode is forced to terminate
# # ppo_config["lr"] = 1e-3
# # ppo_config["create_env_on_driver"] = True
# ppo_config2.update(ppo_config)

# trainer = ppo.PPOTrainer(env="LSIEnv", config=ppo_config)

# Can optionally call trainer.restore(path) to load a checkpoint.

# keys = ["episode_reward_min", "episode_reward_max", "episode_reward_mean", "episodes_this_iter", "episodes_total"]
#
# n = 100
# for i in trange(n):
#     # Perform one iteration of training the policy with PPO
#     result = trainer.train()
#
#     string = ""
#     for key in keys:
#         string += key + f": {result[key]:.2f}, "
#
#     print(string)
#
#     # print(pretty_print(result))
#
#     if (i + 1) % n == 0:
#         checkpoint = trainer.save()
#         print("checkpoint saved at", checkpoint)
#
# trainer.evaluate()

# -------OLD Optimisation--------
# mu = 3
# R = 2000
#
# from ray.tune.suggest.bohb import TuneBOHB
# algo = TuneBOHB(metric="episode_reward_mean", mode="max")
# bohb = ray.tune.schedulers.HyperBandForBOHB(
#     time_attr="training_iteration",
#     metric="episode_reward_mean",
#     mode="max",
#     reduction_factor=mu,
#     max_t=R)
#
# num_samples = 0
# list_n = []
# for s in reversed(range(bohb._s_max_1-1)):
#     n = bohb.get_n0(s)
#     for i in range(s + 1):
#         num_samples += n
#         list_n.append(n)
#         n /= mu
#         n = int(np.ceil(n))
# print(list_n)

# -------------------------------

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch

hebo = HEBOSearch(metric="episode_reward_mean", mode="max",)
hebo = tune.suggest.ConcurrencyLimiter(hebo, max_concurrent=8)

# parameters = [
#     {"name": "cooling_param", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"},
#     {"name": "gain_param", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"},
# ]
# ax_search = AxSearch(space=parameters, metric="episode_reward_mean", mode="max",)
# ax_search = AxSearch(metric="episode_reward_mean", mode="max",)
# ax_search = tune.suggest.ConcurrencyLimiter(ax_search, max_concurrent=6)

# AsyncHyperBand enables aggressive early stopping of bad trials.
scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    grace_period=2,
    max_t=2000,
)

print(scheduler.debug_string())
bracket = scheduler._brackets[0]
print(bracket.cutoff({str(i): i for i in range(20)}))
# tune_parameters = {"cooling_param": tune.uniform(0, 1), "gain_param": tune.uniform(0, 1)}

# trainer = ppo.PPOTrainer(env="LSIEnv", config=ppo_config)

ray.init(num_cpus=8)

tune.run(
    "PPO",
    # stop={"episode_reward_mean": -2,
    #       "training_iteration": 25},
    config=ppo_config,
    num_samples=2000,  # number of trials to perform
    scheduler=scheduler,
    search_alg=hebo,
    local_dir="./outputs/checkpoints",
    checkpoint_at_end=True,
    checkpoint_score_attr="episode_reward_mean",
    keep_checkpoints_num=5,
    # reuse_actors=True
    sync_config=tune.SyncConfig(),
    verbose=2
    # Verbosity mode. 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results. Defaults to 3.
    # fail_fast="raise",

    # resume="AUTO",  # resume from the last run specified in sync_config
)

ray.shutdown()
