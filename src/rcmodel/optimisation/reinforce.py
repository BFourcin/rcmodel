import gym
from gym import spaces

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from xitorch.interpolate import Interp1D
from tqdm.auto import tqdm, trange
import time


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        n = 10
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, out_dim),
        )

        self.log_probs = None  # initialised in on_policy_reset()
        self.on_policy_reset()

    def forward(self, state):
        logits = self.linear_relu_stack(state)

        return logits

    def get_action(self, state):

        # state = self.inputs_to_state(x, unix_time)

        prob_dist = torch.distributions.categorical.Categorical(logits=self.forward(state))  # make a probability distribution

        action = prob_dist.sample()  # sample from distribution pi(a|s) (action given state)

        self.log_probs.append(prob_dist.log_prob(action))  # store log probability of action

        return action

    def on_policy_reset(self):
        # this stores log_probs during an integration step.
        self.log_probs = []
        # self.rewards = []


class Reinforce:
    def __init__(self, env, gamma=0.99, alpha=1e-3):

        self.env = env
        # self.pi = pi
        self.gamma = gamma
        self.alpha = alpha

        self.optimiser = torch.optim.Adam(self.env.RC.cooling_policy.parameters(), lr=self.alpha)

    def update_policy(self, rewards, log_probs):
        log_probs = torch.stack(log_probs).squeeze()  # get into right format regardless of whether single or multiple states have been used

        # downsample rewards by averaging each window.
        rewards = torch.tensor([window.mean() for window in torch.tensor_split(rewards, len(log_probs))])

        # Calculate Discounted Reward:
        discounted_rewards = torch.zeros(len(rewards))

        R = 0
        indx = len(rewards) - 1
        for r in reversed(rewards):  # Future/Later rewards are discounted
            R = r + self.gamma * R  # Discounted Reward is calculated from last reward to first.

            discounted_rewards[indx] = R  # Fill array back to front to un-reverse the order
            indx -= 1

        # Normalise rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)

        # discounted_rewards = torch.tensor(np.array(discounted_rewards.detach().numpy()))

        expected_reward = -log_probs * discounted_rewards  # negative for maximising
        expected_reward = torch.sum(expected_reward)
        # print(f'ER {expected_reward}')

        # Update parameters in pi
        self.optimiser.zero_grad()
        expected_reward.backward()
        self.optimiser.step()

        # print(list(self.pi.parameters())[0].grad)  # check on grads if needed

        return expected_reward

    def train(self, num_episodes):
        self.env.RC.cooling_policy.train()  # Put in training mode
        total_ER = []
        total_rewards = []

        # with tqdm(total=len(self.env.time_data) * num_episodes, position=0, leave=False) as pbar:  # progress bar
        for episode in range(num_episodes):
            self.env.reset()

            episode_rewards = []
            episode_ER = []

            # Time is increased in steps, with the policy updating after every step.
            while self.env.t_index < len(self.env.time_data) - 1:

                # takes a step_size forward in time
                pred, reward = self.env.step(action)  # state and action produced in step

                # Do gradient decent on sample
                ER = self.update_policy(reward, self.env.RC.cooling_policy.log_probs)

                self.env.RC.cooling_policy.on_policy_reset()  # empty buffer

                # get last output and use for next initial value
                # self.env.RC.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

                episode_rewards.append(sum(reward))
                episode_ER.append(ER)

                self.env.t_index += int(step_size)  # increase environment time

            # print(f'Episode {episode+1}, Expected Reward: {sum(episode_ER).item():.2f}, total_reward: {sum(episode_rewards).item():.2f}')

            total_ER.append(sum(episode_ER).detach())
            total_rewards.append(sum(episode_rewards).detach())

        return total_rewards, total_ER


class LSIEnv(gym.Env):
    """Custom Environment that follows gym interface

    config = {"RC_model": rcmodel Class,
              "dataloader": torch.dataloader Object
              }

    A dataloader is used to provide batches of time and temperature data to the model. The environment is run in
    steps of size step_size (15 mins default) with a fixed action, the steps will step through a batch of data until
    it is finished. This is one trajectory and we return the (observation, reward, done, info). At this point we
    expect self.reset() to be called to get the environment ready for the next step.

    The next step will then be a new trajectory and be from the next batch in the dataloader. Once all the batches have
    been seen we refresh the dataloader and go again from the start.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()

        self.RC = config["RC_model"]  # RCModel Class
        self.dataloader = config["dataloader"]
        self.enum_data = None  # enumerate(dataloader)
        self.batch_idx = len(self.dataloader) - 1  # Keeps track of batch number in dataloader, initialised in reset()
        self.epochs = -1  # keeps count of the total epochs of data seen.

        # get dt of data:
        t = self.dataloader.dataset[0][0]
        self.dt = int((t[1]-t[0]).item())

        self.step_size = int((15 * 60)/self.dt)  # num rows of data needed for 15 minutes.
        self.t_index = 0  # used to keep track of index through timeseries
        self.loss_fn = torch.nn.MSELoss()

        # ----- GYM Stuff -----
        self.n_rooms = len(self.RC.building.rooms)

        time_low = [0]
        time_high = [np.float32(np.inf)]

        temp_low = [-10] * (self.n_rooms + 2)  # +2 accounts for the latent nodes
        temp_high = [50] * (self.n_rooms + 2)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2, )

        # Observation is temperature of each room.
        self.observation_space = spaces.Box(np.array(time_low+temp_low), np.array(time_high+temp_high),
                                            dtype=np.float64)

    def step(self, action):
        # Execute a chunk of timeseries
        # t0 = self.observation[0]
        # tn = t0 + (self.step_size+1)*self.dt
        # t_eval = torch.arange(t0, tn, self.dt, dtype=torch.float64)
        t_eval = self.time_data[self.t_index:int(self.t_index + self.step_size)]
        temp = self.temp_data[self.t_index:int(self.t_index + self.step_size), 0:self.n_rooms]

        self.t_index += self.step_size

        # set iv from observation
        self.RC.iv = self.observation[1:].unsqueeze(0).T

        pred = self.RC(t_eval, action)
        pred = pred.squeeze()

        # negative so reward can be maximised
        reward = -self.loss_fn(pred[:, 2:], temp)  # Squared Error

        self.observation = torch.concat((t_eval[-1].unsqueeze(0), pred[-1].detach().clone()))

        # self.state = self._inputs_to_state(pred[-1, 2:], t_eval[-1])  # f(x,t) where: x = [Trm1_tn, Trm2_tn, ...]

        if t_eval[-1] == self.time_data[-1]:  # if this condition is missed then error
            done = True
        else:
            done = False

        return self.observation.numpy(), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.t_index = 0

        # check if we have reached end of data and need to reset enumerate.
        if self.batch_idx + 1 == len(self.dataloader):
            self.enum_data = enumerate(self.dataloader)
            self.epochs += 1

        # get next batch from dataloader:
        self.batch_idx, (self.time_data, self.temp_data) = next(self.enum_data)
        self.time_data = self.time_data.squeeze(0)
        self.temp_data = self.temp_data.squeeze(0)

        # Find correct initial value for current start
        self.RC.iv = self.RC.iv_array(self.time_data[0]).unsqueeze(0).T

        # self.observation = np.array([self.time_data[0].tolist()] + self.RC.iv.T.squeeze(0).tolist())
        self.observation = torch.concat((self.time_data[0].unsqueeze(0), self.RC.iv.T.squeeze(0)))

        return self.observation.numpy()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        return


class Preprocess:
    """
    Class used to preprocess the outputs from the environment before being seen by the policy.
    """
    def __init__(self, mu, std_dev):
        self.mu = mu
        self.std_dev = std_dev

    def preprocess_observation(self, observation):
        """
        Function to transform observation to the state we want the policy to see/use.

        Used to wrap the original environment:
        env = gym.wrappers.TransformObservation(env, preprocess_observation)
        """
        unix_time = observation[0]

        x = observation[3:]  # remove the latent nodes

        # normalise x using info obtained from data.
        x_norm = (x - self.mu) / self.std_dev

        day = 24 * 60 ** 2
        week = 7 * day
        # year = (365.2425) * day

        # state = np.stack([x_norm,
        #                  np.sin(unix_time * (2 * np.pi / day)),
        #                  np.cos(unix_time * (2 * np.pi / day)),
        #                  np.sin(unix_time * (2 * np.pi / week)),
        #                  np.cos(unix_time * (2 * np.pi / week))]).T

        state = x_norm.tolist() + [np.sin(unix_time * (2 * np.pi / day)),
                                   np.cos(unix_time * (2 * np.pi / day)),
                                   np.sin(unix_time * (2 * np.pi / week)),
                                   np.cos(unix_time * (2 * np.pi / week))]

        return state

# def _inputs_to_state(self, x, unix_time):
#     """
#     Convenience function to transform raw input from model into the NN state.
#
#     x: torch.tensor,
#         Tensor of temperatures from the model.
#
#     unix_time: float,
#         Unix epoch time in seconds.
#     """
#     x = x.detach()
#
#     if x.ndim == 0:
#         x = x.unsqueeze(0)
#     if unix_time.ndim == 0:
#         unix_time = unix_time.unsqueeze(0)
#
#     # normalise x using info obtained from data.
#     x_norm = (x - self.mu) / self.std_dev
#     # x_norm = torch.reshape(x_norm, (1,))
#
#     day = 24 * 60 ** 2
#     week = 7 * day
#     # year = (365.2425) * day
#
#     state = torch.stack([torch.flatten(x_norm),
#                         np.sin(unix_time * (2 * np.pi / day)),
#                         np.cos(unix_time * (2 * np.pi / day)),
#                         np.sin(unix_time * (2 * np.pi / week)),
#                         np.cos(unix_time * (2 * np.pi / week))]).to(torch.float32).T
#     # state = state.to(torch.float32)
#
#     # Adds broadcasting to function.
#     # if len(unix_time) > 1:
#     #     state = state.T
#
#     # except TypeError:  # Occurs when no len()
#     #     pass
#     #
#     # try:
#     #     state = torch.cat((x_norm, state), dim=1)
#     #
#     # except RuntimeError:
#     #     # Occurs when single time is being simulated
#     #     state = torch.cat((x_norm, state.unsqueeze(0)), dim=1)
#
#     return state


class PriorEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, RC, time_data, temp_data, prior):
        super().__init__()

        self.RC = RC  # RCModel Class
        self.time_data = time_data  # timeseries
        self.temp_data = temp_data  # temp at timeseries
        self.prior = prior
        self.t_index = 0  # used to keep track of index through timeseries
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        print('Getting actions from prior policy.')
        prior_data = []
        for i in range(len(temp_data)):
            action, _ = prior.get_action(temp_data[i], time_data[i])
            prior_data.append(action)

        self.prior_data = torch.tensor(prior_data).unsqueeze(0).T  # Data can be reused each epoch.

        # ----- GYM Stuff -----
        self.n_rooms = len(self.RC.building.rooms)
        self.low_state = -10
        self.high_state = 50
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2, )

        # Observation is temperature of each room.
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(self.n_rooms,),
            dtype=np.float32
        )

    #                  action
    def step(self, step_size):
        # Execute a chunk of timeseries
        t_eval = self.time_data[self.t_index:int(self.t_index + step_size)]
        dummy_temp = self.temp_data[self.t_index:int(self.t_index + step_size)]

        policy_action, log_prob = self.RC.cooling_policy.get_action(dummy_temp, t_eval)  # Do them all with broadcasting

        if self.RC.cooling_policy.training:  # if in training mode store log_prob
            self.RC.cooling_policy.log_probs.append(log_prob)

        prior_actions = self.prior_data[self.t_index:int(self.t_index + step_size), 0:self.n_rooms]  # read in
        policy_actions = policy_action.unsqueeze(0).T

        # negative so reward can be maximised
        reward = -self.loss_fn(policy_actions, prior_actions)  # Squared Error

        pred = torch.zeros((1,5))  # Fake pred to be compatible with Reinforce
        return pred, reward  # No need for grad on pred
    #          (observation, reward, done, info)
    # self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.t_index = 0
        self.RC.reset_iv()

        self.RC.cooling_policy.on_policy_reset()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        return

