import torch
import pandas as pd
import os
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP
import ray
from ray.rllib.models.preprocessors import NoPreprocessor, get_preprocessor
from ray.rllib.algorithms import Algorithm

import rcmodel.tools

"""
Note: The environment is likely to be wrapped. 
We can still set environment variables using: env.unwrapped.variable = value
Then env.variable will still return value from the base environment.

However if we do env.variable = value, this will only set the variable in the wrapper.
"""


def make_update_env_fn(config):
    """Little function to enable interaction with the environment within Algorithm across all workers.
    >>> algo.workers.foreach_worker(make_update_env_fn(env_config))"""

    def update_env_conf(env):
        env.unwrapped.config.update(config)
        env.unwrapped.update_from_config()  # Do the config update on each worker
        # Recalculate the latent variables.
        env.unwrapped.RC.setup(env.unwrapped.dataloader.dataset)

    def update_env_fn(worker):
        worker.foreach_env(update_env_conf)

    return update_env_fn


# TODO: Train and test need to allow for batches of data to be run, then the batch can
#  be the whole epoch and the model can be updated after each batch.
def train(env, rl_algorithm, optimizer):
    """
    Performs one batch of training.
    Order of rooms in building and in data must match otherwise model will fit wrong rooms to data.
    """
    # model = env.RC

    # Check if DDP
    # if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
    #     ddp_model = model
    #     model = ddp_model.module

    # Must rebuild A & B matrices for each update.
    env.RC.setup()

    env.RC.train()

    terminated = False
    episode_reward = 0
    env.unwrapped.collect_rc_grad = True
    obs, _ = env.reset()
    while not terminated:
        action = rl_algorithm.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

    # Backpropagation
    optimizer.zero_grad()
    episode_reward.backward()
    optimizer.step()

    return episode_reward.item()


def test(env, rl_algorithm, test_dataloader):
    """
    Test the performance of a reinforcement learning algorithm on an environment
    using a given data loader for one epoch.

    Parameters
    ----------
    env : gym.Env
        The environment to test the RL algorithm on.
    rl_algorithm : object
        The RL algorithm to use for testing, which should have a `compute_single_action()`
        method that takes an observation and returns an action.
    test_dataloader : torch.utils.data.DataLoader
        A data loader that provides the testing data as batches of observations and
        their corresponding targets.

    Returns
    -------
    reward_list : list
        A list of episode rewards obtained during testing.
    render_list : list
        A list of rendered frames of the environment during testing.
    """

    base_env = env.unwrapped  # Env is likely wrapped.
    # Use config to update environment, changes are checked. It's weird.
    base_env.config.update(dataloader=test_dataloader)
    base_env.RC.setup(test_dataloader.dataset)  # Get iv array

    # Check if DDP
    # if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
    #     ddp_model = model
    #     model = ddp_model.module

    env.RC.eval()  # Put model in evaluation mode

    reward_list = []
    render_list = []
    with torch.no_grad():
        for i in range(len(test_dataloader)):
            terminated = False
            episode_reward = 0
            obs, _ = env.reset()
            while not terminated:
                action = rl_algorithm.compute_single_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

            reward_list.append(episode_reward)
            render_list += env.render()

    # remove the None values when no render was returned.
    render_list = list(filter(lambda item: item is not None, render_list))
    return reward_list, render_list


# TODO: Add torch lightning support
class OptimiseRC:
    """
    Parameters
    ----------
    model : object
        RCModel class object.
    csv_path : string
        Path to .csv file containing room temperature data.
        Data will be sorted if not done already and saved to a new file with the tag '_sorted'
    sample_size : int
        Length of indexes to sample from dataset per batch.
    dt : int
        Timestep data will be resampled to.
    lr : float
        Learning rate for optimiser.
    model_id : int
        Unique identifier used when optimising multiple models.

    see https://docs.ray.io/en/latest/using-ray-with-pytorch.html
    """

    def __init__(
            self,
            env_config,
            rl_algorithm,
            train_dataset,
            test_dataset,
            lr=1e-3,
            opt_id=0,
    ):
        self.env = rcmodel.tools.env_creator(env_config)
        self.rl_algorithm = rl_algorithm
        self.lr = lr
        self.environment_steps = 0

        self.model_id = opt_id
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=rcmodel.tools.InfiniteSampler(train_dataset),
            batch_size=1,
            shuffle=False,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            sampler=rcmodel.tools.InfiniteSampler(test_dataset),
            batch_size=1,
            shuffle=False,
        )
        self.optimizer = torch.optim.Adam(
            [self.env.RC.params, self.env.RC.loads], lr=lr, maximize=True
        )

        # Check that we don't need preprocessing.
        prep = get_preprocessor(self.env.observation_space)
        assert prep.__name__ == "NoPreprocessor", "Not implemented preprocessing."

    def train(self):
        """Train the model for one epoch."""
        base_env = self.env.unwrapped  # Env is likely wrapped.
        # Use config to update environment, changes are checked. It's weird.
        base_env.config.update(dataloader=self.train_dataloader)
        base_env.RC.setup(self.train_dataloader.dataset)  # Get iv array

        reward_list = []
        render_list = []
        for batch in trange(len(self.train_dataloader), desc='Physical Episodes'):
            reward = train(self.env, self.rl_algorithm, self.optimizer)
            reward_list.append(reward)
            render_list += self.env.render()
            self.environment_steps += self.env.step_count

        # remove the None values when no render was returned.
        render_list = list(filter(lambda item: item is not None, render_list))
        return reward_list, render_list


class OptimisePolicy:
    """
    Parameters
    ----------
    model : object
        RCModel class object.
    csv_path : string
        Path to .csv file containing room temperature data.
        Data will be sorted if not done already and saved to a new file with the tag '_sorted'
    sample_size : int
        Length of indexes to sample from dataset per batch.
    dt : int
        Timestep data will be resampled to.
    lr : float
        Learning rate for optimiser.
    model_id : int
        Unique identifier used when optimising multiple models.

    see https://docs.ray.io/en/latest/using-ray-with-pytorch.html
    """

    def __init__(self, ppo_config, policy_weights=None, opt_id=0):
        # We need to build the algorithm then set the weights on each worker
        self.rl_algorithm = ppo_config.build()
        # Set additional parameters
        self.model_id = opt_id
        self.environment_steps = 0

        if policy_weights:
            weights = ray.put({"default_policy": policy_weights})
            # ... so that we can broadcast these weights to all rollout-workers once.
            for w in self.rl_algorithm.workers.remote_workers():
                w.set_weights.remote(weights)

    def train(self):
        results = self.rl_algorithm.train()
        self.environment_steps += results['num_steps_trained_this_iter']
        avg_reward = results["episode_reward_mean"]
        return avg_reward

    def update_environment(self, config):
        self.rl_algorithm.workers.foreach_worker(make_update_env_fn(config))

    def get_weights(self):
        return self.rl_algorithm.get_policy().get_weights()
