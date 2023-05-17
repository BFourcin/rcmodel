import pytest
import numpy as np
import tempfile
import torch
import ray
import os
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from rcmodel.optimisation.optimise_models import test as not_a_te_st
from rcmodel import (
    RandomSampleDataset,
    InfiniteSampler,
    env_creator,
    OptimiseRC,
    OptimisePolicy,
    model_creator,
)

torch.set_num_threads(1)
register_env("LSIEnv", env_creator)


@pytest.fixture(scope="function")
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def get_datasets():
    csv_path = os.path.join(os.path.dirname(__file__), "data/6hrs_testing_data.csv")
    dt = 30  # seconds
    sample_size = 1 * 60**2 / dt  # ONE HOUR
    warmup_size = 0
    train_dataset = RandomSampleDataset(
        csv_path, sample_size, warmup_size, train=True, test=False
    )
    test_dataset = RandomSampleDataset(
        csv_path, sample_size, warmup_size, train=False, test=True
    )
    return train_dataset, test_dataset


@pytest.fixture
def get_env_config(get_datasets, model_n9):
    train_dataset, test_dataset = get_datasets

    env_config = {
        "RC_model": model_n9,
        "model_pickle_path": None,
        "update_state_dict": None,
        "dataloader": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            sampler=InfiniteSampler(train_dataset),
        ),
        "step_length": 15,  # minutes passed in each step.
        "render_mode": None,  # "single_rgb_array"
    }

    return train_dataset, test_dataset, env_config


@pytest.fixture
def get_policy_config(get_env_config):
    train_dataset, test_dataset, env_config = get_env_config
    n_workers = 1

    policy_config = (
        PPOConfig()
        .training(train_batch_size=12 * n_workers, sgd_minibatch_size=12 * n_workers)
        .environment(env="LSIEnv", env_config=env_config, disable_env_checking=True)
        .rollouts(num_rollout_workers=n_workers, rollout_fragment_length=12 * n_workers)
        .framework(framework="torch")
    )
    return train_dataset, test_dataset, env_config, policy_config


@pytest.fixture
def setup_test(get_policy_config, tmp_dir):
    train_dataset, test_dataset, env_config, policy_config = get_policy_config
    tmpdirname = tmp_dir
    return train_dataset, test_dataset, env_config, policy_config, tmpdirname


def make_first_checkpoint(policy_config, tmpdirname):
    # Build the RL policy
    algo_ppo = policy_config.build()

    # Immediately save as RL Algorithm is rebuilt from checkpoint.
    checkpoint_path = algo_ppo.save(tmpdirname)
    weights = algo_ppo.get_policy().get_weights()
    algo_ppo.stop()
    return checkpoint_path, weights


def test_physical_optimiser(setup_test):
    """Test that we can train a physical optimiser."""
    train_dataset, test_dataset, env_config, policy_config, tmpdirname = setup_test

    # ppo_checkpoint_path, _ = make_first_checkpoint(policy_config, tmpdirname)
    rl_algorithm = policy_config.build()

    op = OptimiseRC(
        env_config, rl_algorithm, train_dataset, test_dataset, lr=1e-3, opt_id=0
    )

    # unwrap the render wrapper because it breakes the test as render not set up for
    # more than 1 room.
    # op.env = op.env.env

    op.train()
    not_a_te_st(op.env, rl_algorithm, op.test_dataloader)


def test_policy_optimiser(setup_test):
    """Test that we can train a policy optimiser."""
    _, _, _, policy_config, tmpdirname = setup_test
    _, weights = make_first_checkpoint(policy_config, tmpdirname)

    op_pol = OptimisePolicy(policy_config, weights, opt_id=0)

    # A bit of a hacky way to call rcmodel.setup()
    op_pol.update_environment({})

    op_pol.train()


def test_policy_env_update(setup_test):
    """Test that we can update the environment of the policy optimiser while it's running.
    We
    """
    _, _, env_config, policy_config, tmpdirname = setup_test

    _, weights = make_first_checkpoint(policy_config, tmpdirname)

    # Worker class so we can throw policy optimiser into a Ray remote object and
    # interact with it while it's running.
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
    original_state_dict = env_config["RC_model"].state_dict().copy()

    # Change the state dict to anything, we just make it equal one.
    for key, value in original_state_dict.items():
        original_state_dict[key] = value / value

    env_config["update_state_dict"] = original_state_dict

    # Update the environment with our state dict
    [w.update_environment.remote(env_config) for w in workers]

    # Initialise training
    [w.train.remote() for w in workers]

    # while training grab the state dict from the environment
    env_info = [ray.get(w.get_info.remote()) for w in workers]
    new_state_dict = env_info[0][1][0]

    for key in original_state_dict:
        assert torch.equal(original_state_dict[key], new_state_dict[key])


# add test for saving and loading.

# add test for cycle.

# add test to make sure images are being saved and the different options work.

# test that model parameters are changed when a cycle occurs.

if __name__ == "__main__":
    pytest.main()
