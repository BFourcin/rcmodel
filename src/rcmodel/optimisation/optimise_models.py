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

    done = False
    episode_reward = 0
    env.unwrapped.collect_rc_grad = True
    obs = env.reset()
    while not done:
        action = rl_algorithm.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
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
            done = False
            episode_reward = 0
            obs = env.reset()
            while not done:
                action = rl_algorithm.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            reward_list.append(episode_reward)
            render_list += env.render(env.render_mode)

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
            render_list += self.env.render(self.env.render_mode)

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

        if policy_weights:
            weights = ray.put({"default_policy": policy_weights})
            # ... so that we can broadcast these weights to all rollout-workers once.
            for w in self.rl_algorithm.workers.remote_workers():
                w.set_weights.remote(weights)

    def train(self):
        results = self.rl_algorithm.train()
        avg_reward = results["episode_reward_mean"]
        return avg_reward

    def update_environment(self, config):
        self.rl_algorithm.workers.foreach_worker(make_update_env_fn(config))

    def get_weights(self):
        return self.rl_algorithm.get_policy().get_weights()

# class DDPOptimiseRC:
#     """
#     Parameters
#     ----------
#     model : object
#         RCModel class object.
#     csv_path : string
#         Path to .csv file containing room temperature data.
#         Data will be sorted if not done already and saved to a new file with the tag '_sorted'
#     sample_size : int
#         Length of indexes to sample from dataset per batch.
#     dt : int
#         Timestep data will be resampled to.
#     lr : float
#         Learning rate for optimiser.
#     model_id : int
#         Unique identifier used when optimising multiple models.
#
#     see https://docs.ray.io/en/latest/using-ray-with-pytorch.html
#     """
#
#     def __init__(self, model, train_dataset, test_dataset, lr=1e-3, opt_id=0):
#         self.model = model
#         self.lr = lr
#         self.model_id = opt_id
#
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
#
#     def parallel_loop(self, rank, world_size, func, dataset, q):
#         """
#         Distributed training loop. Splits a dataset into n=world_size batches and runs simultaneously on multiple processes.
#
#         This is the function that is actually ran in each distributed process.
#
#         Note: as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor,
#         you don’t need to worry about different DDP processes start from different model parameter initial values.
#
#         rank: int
#             The index of the process, parent has rank 0.1
#         world_size: int
#             Number of process/cpus to use. This is also the number of batches to split the dataset into.
#         func: function
#             Function to run in parallel. Either train or test.
#         q: mp.queue
#             Multiprocessing object, used to get info back from child processes.
#         """
#         # print()
#         # print(f"Start running DDP with model parallel example on rank: {rank}.")
#         # print(f'current process: {mp.current_process()}')
#         # print(f'pid: {os.getpid()}')
#
#         setup_process(rank, world_size)
#
#         sampler = torch.utils.data.DistributedSampler(
#             dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
#         )
#
#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=1,
#             pin_memory=False,
#             num_workers=0,
#             drop_last=True,
#             shuffle=False,
#             sampler=sampler,
#         )
#
#         # create distributed model
#         ddp_model = DDP(self.model, find_unused_parameters=True)
#
#         if func is train:
#             optimizer = torch.optim.Adam(
#                 [ddp_model.module.params, ddp_model.module.loads], lr=self.lr
#             )
#             loss = func(ddp_model, dataloader, optimizer)
#         elif func is test:
#             loss = func(ddp_model, dataloader)
#
#         # This gathers info from all processes.
#         all_losses = [torch.zeros(1, dtype=torch.float32) for _ in range(world_size)]
#         dist.all_gather(all_losses, torch.tensor([loss], dtype=torch.float32))
#
#         # put the gathered loss onto our mp.queue object.
#         if rank == 0:
#             avg_loss = torch.tensor(all_losses).mean()  # avg loss across all processes.
#             q.put(avg_loss)
#
#         # Destroy a given process group, and de-initialise the distributed package
#         cleanup()
#
#     def run_parallel_loop(self, func, dataset):
#         torch.set_num_threads(1)
#
#         world_size = len(dataset) if len(dataset) <= mp.cpu_count() else mp.cpu_count()
#
#         result_queue = mp.Queue()
#         for rank in range(world_size):
#             mp.Process(
#                 target=self.parallel_loop,
#                 args=(rank, world_size, func, dataset, result_queue),
#             ).start()
#
#         avg_loss = result_queue.get()
#
#         return avg_loss
#
#     def train(self):
#         avg_loss = self.run_parallel_loop(train, self.train_dataset)
#         return avg_loss
#
#     def test(self):
#         avg_loss = self.run_parallel_loop(test, self.test_dataset)
#         return avg_loss
#
#
# def setup_process(rank, world_size, backend="gloo"):
#     """
#     Initialize the distributed environment (for each process).
#
#     gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
#     it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.
#     """
#     # set up the master's ip address so this child process can coordinate
#     # os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"  # '12355'
#
#     # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
#     # if torch.cuda.is_available():
#     #     backend = 'nccl'
#     # Initializes the default distributed process group, and this will also initialize the distributed package.
#     dist.init_process_group(
#         backend,
#         timeout=datetime.timedelta(seconds=10),
#         rank=rank,
#         world_size=world_size,
#     )
#
#
# def cleanup():
#     """Destroy a given process group, and deinitialize the distributed package"""
#     dist.destroy_process_group()
