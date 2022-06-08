import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pathlib import Path
from tqdm.auto import tqdm, trange
import cma
import pickle

from rcmodel import *

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

# Load model?
load_model_path_physical = './prior_physical.pt'  # or None
load_model_path_policy = './prior_policy.pt'


# load_model_path_physical = './physical_model.pt'  # or None
# load_model_path_policy = './prior_policy.pt'

# opt_id = 0


####################
def replace_parameters(state_dict, new_parameters):
    # new_parameters between 0-1

    count = 0
    for name in list(state_dict)[0:2]:
        # name = list(state_dict)[0]
        layer = state_dict[name]

        n = torch.prod(torch.tensor(layer.size())).item()  # num of parameters in layer

        state_dict[name] = torch.nn.Parameter(
            torch.reshape(torch.tensor(new_parameters[count:count + n], dtype=torch.float32), layer.size()))

        count += n

    return state_dict


@ray.remote
def fitness(params, env, dataset):
    # with FileLock("parameters.csv.lock"):
    #     with open("parameters.csv","a") as f:
    #         np.savetxt(f, [params], delimiter=', ')

    env.RC.load_state_dict(replace_parameters(env.RC.state_dict(), list(params)))

    t, T = dataset.get_all_data()

    env.RC.iv_array = env.RC.get_iv_array(t)

    with torch.no_grad():
        total_reward = 0
        for i in range(len(dataset)):
            done = False
            observation = env.reset()
            while not done:
                # env.render()
                action = env.RC.cooling_policy.get_action(torch.tensor(observation, dtype=torch.float32)).item()
                observation, reward, done, _ = env.step(action)
                total_reward += -reward

        return total_reward  # cmaes minimises


####################
def worker():
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
        "load_model_path_policy": load_model_path_policy,  # or None
        "load_model_path_physical": load_model_path_physical,  # or None
    }

    # Initialise Model
    # model = model_creator(model_config)

    # # Initialise Optimise Class - for training physical model
    dt = 30
    sample_size = 24 * 60 ** 2 / dt  # ONE DAY
    # warmup_size = sample_size * 7  # ONE WEEK
    warmup_size = 0
    train_dataset = BuildingTemperatureDataset(model_config['room_data_path'], sample_size, all=True)


    config = {
        "env": LSIEnv,
        "env_config": {"model_config": model_config,
                       "dataloader": torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False),
                       "step_length": 15  # minutes passed in each step.
                       },

        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_workers": 8,  # parallelism
        "framework": "torch",
    }

    def env_creator(env_config, model=None):
        if model is None:
            model_config = env_config["model_config"]
            model = model_creator(model_config)
            time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1],
                                     dtype=torch.float64)
            model.iv_array = model.get_iv_array(time_data)

        env_config["RC_model"] = model

        env = LSIEnv(env_config)
        # wrap environment:
        env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

        return env

    env = env_creator(config["env_config"])

    # physical training:
    x0 = env.RC.params.tolist() + env.RC.loads.flatten().tolist()

    result = cmaes(env, train_dataset, x0)

    best_params = result[0].tolist()
    env.RC.load_state_dict(replace_parameters(env.RC.state_dict(), list(best_params)))

    model_id = 1
    env.RC.save(model_id)

    # Policy Training:

    # def dummy_env_creator(dummy_config): # passes env straight to ppo
    #     return dummy_config["env"]

    register_env("LSIEnv", env_creator)

    # ray.init()
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config["vf_clip_param"] = 3000
    ppo_config["train_batch_size"] = 10
    ppo_config["sgd_minibatch_size"] = 10
    ppo_config["lr"] = 1e-3

    model_config["load_model_path_policy"] = f'.outputs/rcmodel{model_id}.pt'
    model_config["load_model_path_physical"] = f'.outputs/rcmodel{model_id}.pt'
    train_dataset = BuildingTemperatureDataset(model_config['room_data_path'], sample_size, all=True)
    config["env_config"]["dataloader"] = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    ppo_config.update(config)

    # error because env already exists, maybe del env?
    trainer = ppo.PPOTrainer(env="LSIEnv", config=ppo_config)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    keys = ["episode_reward_min", "episode_reward_max", "episode_reward_mean", "episodes_this_iter", "episodes_total"]

    for i in trange(10):
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


@ray.remote
def get_results(env, dataset, X):
    return ray.get([fitness.remote(params, env, dataset) for params in X])


def cmaes(env, dataset, x0):
    # hyper parameters:
    n_workers = 2  # multiprocessing
    # maxfevals = n_workers*150
    maxfevals = 1

    sigma0 = 1.5
    opts = {
        # 'bounds': [0, 1],
        'maxfevals': maxfevals,
        'tolfun': 1e-6,
        'popsize': n_workers
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # es.optimize(fitness, args=(model, t_evals, temp), maxfun=2)
    # res = es.result
    # print(res)

    while not es.stop():
        X = es.ask()
        es.tell(X, ray.get(get_results.remote(env, dataset, X)))
        es.disp()
        es.logger.add()

    # save the run:
    s = es.pickle_dumps()  # return pickle.dumps(es) with safeguards
    # save string s to file like open(filename, 'wb').write(s)
    open('savedrun', 'wb').write(s)

    return es.best.get()


if __name__ == '__main__':
    worker()
