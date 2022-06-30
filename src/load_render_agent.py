from ray.tune import ExperimentAnalysis
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
import pandas as pd
import ray
import torch

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


# best_config = analysis.get_best_config("episode_reward_mean", mode='max')
# best_config["num_workers"] = 1
#
# checkpoint_path = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', 'max'),
#                                                    metric='episode_reward_mean')
# print(checkpoint_path)

# --------------------------

def func_iv_array(model_config):
    mc = model_config.copy()
    mc["cooling_param"], mc["gain_param"] = None, None
    model = model_creator(mc)
    time_data = torch.tensor(pd.read_csv(mc['room_data_path'], skiprows=0).iloc[:, 1], dtype=torch.float64)
    return model.get_iv_array(time_data)

weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

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
   'room_data_path': csv_path,
   'load_model_path_policy': None,
   'load_model_path_physical': 'trained_model.pt',
   'cooling_param': 0.09133423646610082,
   'gain_param': 0.9086668150306394},
  'dataloader': None,
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
best_config['env_config']['iv_array'] = func_iv_array(best_config['env_config']['model_config'])
dt = 30
sample_size = 24 * 60 ** 2 / dt  # ONE DAY
# warmup_size = sample_size * 7  # ONE WEEK
warmup_size = 0
train_dataset = RandomSampleDataset(best_config['env_config']['model_config']['room_data_path'], sample_size, warmup_size, train=True, test=False)
best_config['env_config']['dataloader'] = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)



# ------------------------------------------
register_env("LSIEnv", env_creator)

agent = ppo.PPOTrainer(env="LSIEnv", config=best_config)
# agent.restore(checkpoint_path[0][0])

checkpoint_path = '/Users/benfourcin/Documents/Github/rcmodel/src/outputs/checkpoints/PPO/PPO_LSIEnv_38dd8635_91_env=LSIEnv,dataloader=torch_utils_data_dataloader_DataLoader_object_at_0x7fd9d210ddc0,iv_array=xitorch_inte_2022-06-23_06-21-14/checkpoint_002000/checkpoint-2000'
agent.restore(checkpoint_path)

# def load_weights(path):
#     import pickle
#     return pickle.load(open(path, "rb"))
#
#
# trained_weights = load_weights('./trained_weights.pkl')
# agent.set_weights(trained_weights)
import matplotlib.pyplot as plt

env = env_creator(best_config["env_config"])
print("env created")
for i in range(5):
    print(i)
    done = False
    observation = env.reset()
    tot_reward = 0
    while not done:
        im = env.render('rgb_array')
        print(im)
        if im is not None:
            if len(im) > 0:
                plt.imshow(im)
                plt.show()
        # action = np.random.randint(2)
        action = agent.compute_single_action(observation)
        observation, reward, done, _ = env.step(action)
        tot_reward += reward
