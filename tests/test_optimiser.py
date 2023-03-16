import pytest
import numpy as np
import torch
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from rcmodel import (
    RandomSampleDataset,
    env_creator,
    OptimiseRC,
    OptimisePolicy,
    model_creator,
)

torch.set_num_threads(1)

weather_data_path = "/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv"
csv_path = "/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv"
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
    "room_coordinates": [
        [
            [92.07, 125.94],
            [92.07, 231.74],
            [129.00, 231.74],
            [154.45, 231.74],
            [172.64, 231.74],
            [172.64, 125.94],
        ]
    ],
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
    },
}
dt = 30
sample_size = 24 * 60**2 / dt  # ONE DAY
warmup_size = 0
train_dataset = RandomSampleDataset(
    csv_path, sample_size, warmup_size, train=True, test=False
)
test_dataset = RandomSampleDataset(
    csv_path, sample_size, warmup_size, train=False, test=True
)
env_config = {
    "model_pickle_path": None,
    "model_state_dict": None,
    "model_config": model_config,
    "dataloader": torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    ),
    "step_length": 15,  # minutes passed in each step.
    "render_mode": "single_rgb_array",  # "single_rgb_array"
}
n_workers = 1

register_env("LSIEnv", env_creator)
# Set up Policy algorithm and environment variables
policy_config = (
    PPOConfig()
    .training(train_batch_size=12 * n_workers, sgd_minibatch_size=12 * n_workers)
    .environment(env="LSIEnv", env_config=env_config, disable_env_checking=True)
    .rollouts(num_rollout_workers=n_workers, rollout_fragment_length=12 * n_workers)
    .framework(framework="torch")
)


def make_first_checkpoint():
    # Build the RL policy
    algo_ppo = policy_config.build()

    # Immediately save as RL Algorithm is rebuilt from checkpoint.
    checkpoint_path = algo_ppo.save()
    weights = algo_ppo.get_policy().get_weights()
    algo_ppo.stop()
    return checkpoint_path, weights


def test_physical_optimiser():
    """Test that we can train a physical optimiser."""
    ppo_checkpoint_path, _ = make_first_checkpoint()

    op = OptimiseRC(
        env_config, ppo_checkpoint_path, train_dataset, test_dataset, lr=1e-3, opt_id=0
    )

    op.train()
    op.test()


def test_policy_optimiser():
    """Test that we can train a policy optimiser."""
    _, weights = make_first_checkpoint()

    op_pol = OptimisePolicy(policy_config, weights, opt_id=0)

    op_pol.train()
    # op_pol.test()


def test_policy_env_update():
    """Test that we can update the environment of the policy optimiser while it's running."""
    _, weights = make_first_checkpoint()

    # Worker class so we can throw policy optimiser into a Ray remote object and interact with it while it's running.
    @ray.remote
    class Worker:
        def __init__(self, policy_config, weights):
            self.policy_optimiser = OptimisePolicy(policy_config, weights, opt_id=0)

        def train(self):
            self.policy_optimiser.train()

        def update_environment(self, env_config):
            self.policy_optimiser.update_environment(env_config)

        def get_info(self):
            def make_info_env_fn():
                """Little function to enable interaction with the environment within Algorithm across all workers.
                >>> algo.workers.foreach_worker(make_info_env_fn())"""

                def get_env_info(env):
                    return env.RC.state_dict()

                def look_env_fn(worker):
                    return worker.foreach_env(get_env_info)

                return look_env_fn

            return self.policy_optimiser.rl_algorithm.workers.foreach_worker(
                make_info_env_fn()
            )

    # Initialise workers
    workers = [Worker.remote(policy_config, weights) for _ in range(1)]

    # Make a model so we can get the state dict
    original_state_dict = model_creator(env_config["model_config"]).state_dict()

    # Change the state dict to anything, we just make it equal one.
    for key, value in original_state_dict.items():
        original_state_dict[key] = value / value

    env_config["model_state_dict"] = original_state_dict

    # Update the environment with our state dict
    [w.update_environment.remote(env_config) for w in workers]

    # Initialise training
    [w.train.remote() for w in workers]

    # while training grab the state dict from the environment
    env_info = [ray.get(w.get_info.remote()) for w in workers]
    new_state_dict = env_info[0][1][0]

    for key in original_state_dict:
        assert torch.equal(original_state_dict[key], new_state_dict[key])
