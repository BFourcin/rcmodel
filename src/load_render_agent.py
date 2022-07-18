from ray.tune import ExperimentAnalysis
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
import pandas as pd
import numpy as np
import ray
import torch
import time
from moviepy.editor import ImageSequenceClip
import tempfile
from PIL import Image
# from moviepy.video.io.bindings import mplfig_to_npimage

from rcmodel import *


# ray.init()

# analysis = ExperimentAnalysis('~/Documents/Github/rcmodel/src/outputs/checkpoints/PPO')


def env_creator(env_config):
    with torch.no_grad():
        # global config
        # print(config)
        model_config = env_config["model_config"]
        # print(tune_parameters)
        # model_config["cooling_param"] = env_config["cooling_param"]
        # model_config["gain_param"] = env_config["gain_param"]
        model = model_creator(model_config)
        # time_data = torch.tensor(pd.read_csv(model_config['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
        # model.iv_array = model.get_iv_array(time_data)
        model.iv_array = env_config["iv_array"]

        env_config["RC_model"] = model

        env = LSIEnv(env_config)

        # wrap environment:
        env = PreprocessEnv(env, mu=23.359, std_dev=1.41)

    return env

# --------------------------


def func_iv_array(model_config, dataset):
    mc = model_config.copy()
    mc["cooling_param"], mc["gain_param"] = None, None
    model = model_creator(mc)
    return get_iv_array(model, dataset)


weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/summer2021_sorted.csv'

dt = 30
sample_size = 67 * 24 * 60 ** 2 / dt
# sample_size = 1 * 24 * 60 ** 2 / dt  # ONE DAY
warmup_size = 7 * 24 * 60 ** 2 / dt  # ONE WEEK
# warmup_size = 0

# train_dataloader, test_dataloader = dataloader_creator(csv_path, sample_size, warmup_size)

all_dataset = RandomSampleDataset(csv_path, sample_size, warmup_size, all=True)
all_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=False)

dataloader = all_dataloader

best_config = {'env': 'LSIEnv',
               'env_config': {'model_config': {'C_rm': [1000.0, 100000.0],
                                               'C1': [100000.0, 100000000.0],
                                               'C2': [100000.0, 100000000.0],
                                               'R1': [0.1, 5],
                                               'R2': [0.1, 5],
                                               'R3': [0.5, 6],
                                               'Rin': [0.1, 5],
                                               'cool': [0, 50],
                                               'gain': [0, 5],
                                               'room_names': ['seminar_rm_a_t0106'],
                                               'room_coordinates': [[[92.07, 125.94],
                                                                     [92.07, 231.74],
                                                                     [129.0, 231.74],
                                                                     [154.45, 231.74],
                                                                     [172.64, 231.74],
                                                                     [172.64, 125.94]]],
                                               'weather_data_path': weather_data_path,
                                               'load_model_path_policy': None,
                                               'load_model_path_physical': 'trained_model.pt',
                                               'cooling_param': 0.09133423646610082,
                                               'gain_param': 0.9086668150306394},
                              'dataloader': dataloader,
                              'step_length': 15,
                              'iv_array': None},
               'num_gpus': 0,
               'num_workers': 1,
               'framework': 'torch',
               'vf_clip_param': 3000,
               'rollout_fragment_length': 480,
               'train_batch_size': 1440,
               'sgd_minibatch_size': 96,
               'horizon': None}

start = time.time()
best_config['env_config']['iv_array'] = func_iv_array(best_config['env_config']['model_config'], dataloader.dataset)
print(f'time for IV array: {time.time() - start:.2f} seconds')


print(f'Dataset length: {len(best_config["env_config"]["dataloader"].dataset)}')
# ------------------------------------------
register_env("LSIEnv", env_creator)


agent = ppo.PPOTrainer(env="LSIEnv", config=best_config)
# agent.restore(checkpoint_path[0][0])

# checkpoint_path = '/Users/benfourcin/Documents/Github/rcmodel/src/outputs/checkpoints/PPO/PPO_LSIEnv_38dd8635_91_env=LSIEnv,dataloader=torch_utils_data_dataloader_DataLoader_object_at_0x7fd9d210ddc0,iv_array=xitorch_inte_2022-06-23_06-21-14/checkpoint_002000/checkpoint-2000'
checkpoint_path = '/Users/benfourcin/Desktop/tuned_trial/PPO/PPO_LSIEnv_60f2f_00000_0_2022-06-30_11-14-08/checkpoint_000275/checkpoint-275'

agent.restore(checkpoint_path)


def save_image_from_array(image_arr, tmpdirname, image_num):
    im = Image.fromarray(image_arr)
    im.save(f"{tmpdirname}/frame{image_num}.jpeg")


with tempfile.TemporaryDirectory() as tmpdirname:

    env = env_creator(best_config["env_config"])
    print("env created")
    images_list = []
    image_num = 0
    for i in range(len(best_config['env_config']['dataloader'].dataset)):
        # print(i)
        done = False
        observation = env.reset()
        tot_reward = 0
        while not done:
            save_image_from_array(env.render('rgb_array'), tmpdirname, image_num)
            image_num += 1
            # env.render('human')
            action = agent.compute_single_action(observation)
            observation, reward, done, _ = env.step(action)
            tot_reward += reward
            if done:
                print(f'Reward for episode: {tot_reward:.2f}')

    print('Done!')

    clip = ImageSequenceClip(tmpdirname, fps=25)
    clip.write_videofile('./outputs/trained_model.mp4')

ray.shutdown()
