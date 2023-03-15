import pytest
import numpy as np
import torch
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from rcmodel import RandomSampleDataset, env_creator, OptimiseRC, OptimisePolicy, model_creator

torch.set_num_threads(1)

weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
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
            }
        }
dt = 30
sample_size = 24 * 60 ** 2 / dt  # ONE DAY
warmup_size = 0
train_dataset = RandomSampleDataset(csv_path, sample_size, warmup_size, train=True, test=False)
test_dataset = RandomSampleDataset(csv_path, sample_size, warmup_size, train=False, test=True)
env_config = {"model_pickle_path": None,
              "model_state_dict": None,
              "model_config": model_config,
              "dataloader": torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False),
              "step_length": 15,  # minutes passed in each step.
              "render_mode": 'single_rgb_array',  # "single_rgb_array"
              }
n_workers = 1

register_env("LSIEnv", env_creator)
# Set up Policy algorithm and environment variables
policy_config = PPOConfig().training(train_batch_size=96 * 5 * n_workers) \
    .environment(env="LSIEnv", env_config=env_config, disable_env_checking=True) \
    .rollouts(num_rollout_workers=n_workers, rollout_fragment_length=96 * 5) \
    .framework(framework='torch')

def make_first_checkpoint():

    # Build the RL policy
    algo_ppo = policy_config.build()

    # Immediately save as RL Algorithm is rebuilt from checkpoint.
    checkpoint_path = algo_ppo.save()
    weights = algo_ppo.get_policy().get_weights()
    algo_ppo.stop()
    return checkpoint_path, weights


def test_physical_optimiser():
    ppo_checkpoint_path, _ = make_first_checkpoint()

    op = OptimiseRC(env_config, ppo_checkpoint_path, train_dataset, test_dataset, lr=1e-3, opt_id=0)

    op.train()
    op.test()


def test_policy_optimiser():
    _, weights = make_first_checkpoint()

    op_pol = OptimisePolicy(policy_config, weights, opt_id=0)

    op_pol.train()
    # op_pol.test()


def test_policy_env_update():
    _, weights = make_first_checkpoint()

    op_pol = OptimisePolicy(policy_config, weights, opt_id=0)

    op_pol.train()
    # op_pol.test()

    sd = model_creator(env_config["model_config"]).state_dict()

    for key, value in sd.items():
        sd[key] = value/value

    env_config['model_state_dict'] = sd

    op_pol.update_environment(env_config)

    op_pol.train()









