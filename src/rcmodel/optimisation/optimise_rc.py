import torch
import pandas as pd
import os
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ray.rllib.models.preprocessors import NoPreprocessor, get_preprocessor
from ray.rllib.algorithms import Algorithm


import rcmodel.tools


def sample_action(rl_algorithm, obs):
    """
    Sample action from policy distribution.
    Obs must be pre-processed if needed.
    For more info: https://docs.ray.io/en/latest/rllib/rllib-training.html#accessing-model-state
    """
    # Run a forward pass to get model output logits. Note that complex observations
    # must be preprocessed as in the above code block.
    logits, _ = rl_algorithm.model({"obs": torch.tensor(obs).unsqueeze(0)})

    # Compute action distribution given logits
    dist = rl_algorithm.dist_class(logits, rl_algorithm.model)

    # Query the distribution for samples, sample logp
    action = dist.sample()

    # logp = dist.logp(torch.tensor(1))

    return action


def train(env, policy, optimizer):
    """
    Performs one epoch of training.
    Order of rooms in building and in data must match otherwise model will fit wrong rooms to data.
    """
    model = env.RC

    # Perform integration across entire timeseries to get initial value array
    model.iv_array = rcmodel.tools.get_iv_array(model, env.dataloader.dataset)

    # Check if DDP
    if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
        ddp_model = model
        model = ddp_model.module

    model.train()
    num_batches = len(env.dataloader)

    episode_reward = 0
    done = False
    env.env.collect_rc_grad = True
    obs = env.reset()
    while not done:
        action
        obs, reward, done, info = env.step(action)

        episode_reward += reward / num_batches

    # Backpropagation
    optimizer.zero_grad()
    episode_reward.backward()
    optimizer.step()

    return episode_reward.item()


# def train(model, dataloader, optimizer):
#     """
#     Performs one epoch of training.
#     Order of rooms in building and in data must match otherwise model will fit wrong rooms to data.
#     """
#
#     # Check if DDP
#     if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
#         ddp_model = model
#         model = ddp_model.module
#
#     model.train()
#
#     # if policy exists:
#     if model.cooling_policy:
#         model.cooling_policy.eval()
#
#         # This ended up causing the policy not to optimise:
#         # Stops Autograd endlessly keeping track of the graph. Memory Leak!
#         # try:
#         #     for layer in model.cooling_policy.parameters():
#         #         layer.requires_grad = False
#         #
#         # # This occurs when policy contains no parameters e.g. when using prior to train.
#         # except AttributeError:
#         #     pass
#
#     num_cols = len(model.building.rooms)  # number of columns to use from data.
#     num_batches = len(dataloader)
#     train_loss = 0
#     loss_fn = torch.nn.MSELoss()
#     sum_loss = 0
#
#     for batch, (time, temp) in enumerate(dataloader):
#         # Get model arguments:
#         time = time.squeeze(0)
#         temp = temp.squeeze(0)
#
#         # Compute prediction and loss
#         pred = model(time)
#         pred = pred.squeeze(-1)  # change from column to row matrix
#
#         loss = loss_fn(pred[:, 2:], temp[:, 0:num_cols])
#         sum_loss += loss/num_batches
#
#         # get last output and use for next initial value
#         # model.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD
#
#     # Backpropagation
#     optimizer.zero_grad()
#     sum_loss.backward()
#     optimizer.step()
#
#     return sum_loss.item()


def test(env, rl_algorithm):
    model = env.RC

    # Check if DDP
    if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
        ddp_model = model
        model = ddp_model.module

    model.eval()  # Put model in evaluation mode
    num_batches = len(env.dataloader)

    episode_reward = 0
    done = False
    obs = env.reset()
    with torch.no_grad():
        while not done:
            action = rl_algorithm.compute_single_action(obs)

            obs, reward, done, info = env.step(action)

            episode_reward += reward / num_batches

    return episode_reward.item()


# def test(model, dataloader):
#     # Check if DDP
#     if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
#         ddp_model = model
#         model = ddp_model.module
#
#     model.reset_iv()  # Reset initial value
#     model.eval()  # Put model in evaluation mode
#     num_batches = len(dataloader)
#     num_cols = len(model.building.rooms)  # number of columns to take from data.
#     test_loss = 0
#     loss_fn = torch.nn.MSELoss()
#
#     with torch.no_grad():
#         for (time, temp) in dataloader:
#             time = time.squeeze(0)
#             temp = temp.squeeze(0)
#
#             pred = model(time)
#             pred = pred.squeeze(-1)  # change from column to row matrix
#             test_loss += loss_fn(pred[:, 2:], temp[:, 0:num_cols]).item()
#
#             # get last output and use for next initial value
#             model.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD
#
#     test_loss /= num_batches
#
#     return test_loss


def setup_process(rank, world_size, backend="gloo"):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.
    """
    # set up the master's ip address so this child process can coordinate
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # '12355'

    # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
    # if torch.cuda.is_available():
    #     backend = 'nccl'
    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(
        backend,
        timeout=datetime.timedelta(seconds=10),
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    """Destroy a given process group, and deinitialize the distributed package"""
    dist.destroy_process_group()


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

    def __init__(self, env_config, ppo_checkpoint_path, train_dataset, test_dataset, lr=1e-3, opt_id=0):

        self.env = rcmodel.tools.env_creator(env_config)
        self.rl_algorithm = Algorithm.from_checkpoint(ppo_checkpoint_path)
        self.lr = lr
        # self.model.init_physical()  # randomise parameters - DO WE WANT THIS?
        self.model_id = opt_id
        self.train_dataset = train_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=False
        )
        self.test_dataset = test_dataset
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )

        self.optimizer = torch.optim.Adam(
            [self.env.RC.params, self.env.RC.loads], lr=lr, maximize=True
        )

        # Check that we don't need preprocessing.
        prep = get_preprocessor(self.env.observation_space)
        assert prep.__name__ == 'NoPreprocessor', 'Not implemented preprocessing.'

    def train(self):
        self.env.dataloader = self.train_dataloader
        avg_reward = train(self.env, self.rl_algorithm, self.optimizer)
        return avg_reward

    def test(self):
        self.env.dataloader = self.test_dataloader
        avg_reward = test(self.env, self.rl_algorithm)
        return avg_reward


class DDPOptimiseRC:
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

    def __init__(self, model, train_dataset, test_dataset, lr=1e-3, opt_id=0):
        self.model = model
        self.lr = lr
        self.model_id = opt_id

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def parallel_loop(self, rank, world_size, func, dataset, q):
        """
        Distributed training loop. Splits a dataset into n=world_size batches and runs simultaneously on multiple processes.

        This is the function that is actually ran in each distributed process.

        Note: as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor,
        you don’t need to worry about different DDP processes start from different model parameter initial values.

        rank: int
            The index of the process, parent has rank 0.1
        world_size: int
            Number of process/cpus to use. This is also the number of batches to split the dataset into.
        func: function
            Function to run in parallel. Either train or test.
        q: mp.queue
            Multiprocessing object, used to get info back from child processes.
        """
        # print()
        # print(f"Start running DDP with model parallel example on rank: {rank}.")
        # print(f'current process: {mp.current_process()}')
        # print(f'pid: {os.getpid()}')

        setup_process(rank, world_size)

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            pin_memory=False,
            num_workers=0,
            drop_last=True,
            shuffle=False,
            sampler=sampler,
        )

        # create distributed model
        ddp_model = DDP(self.model, find_unused_parameters=True)

        if func is train:
            optimizer = torch.optim.Adam(
                [ddp_model.module.params, ddp_model.module.loads], lr=self.lr
            )
            loss = func(ddp_model, dataloader, optimizer)
        elif func is test:
            loss = func(ddp_model, dataloader)

        # This gathers info from all processes.
        all_losses = [torch.zeros(1, dtype=torch.float32) for _ in range(world_size)]
        dist.all_gather(all_losses, torch.tensor([loss], dtype=torch.float32))

        # put the gathered loss onto our mp.queue object.
        if rank == 0:
            avg_loss = torch.tensor(all_losses).mean()  # avg loss across all processes.
            q.put(avg_loss)

        # Destroy a given process group, and de-initialise the distributed package
        cleanup()

    def run_parallel_loop(self, func, dataset):
        torch.set_num_threads(1)

        world_size = len(dataset) if len(dataset) <= mp.cpu_count() else mp.cpu_count()

        result_queue = mp.Queue()
        for rank in range(world_size):
            mp.Process(
                target=self.parallel_loop,
                args=(rank, world_size, func, dataset, result_queue),
            ).start()

        avg_loss = result_queue.get()

        return avg_loss

    def train(self):
        avg_loss = self.run_parallel_loop(train, self.train_dataset)
        return avg_loss

    def test(self):
        avg_loss = self.run_parallel_loop(test, self.test_dataset)
        return avg_loss
