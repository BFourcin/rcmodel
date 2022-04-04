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
    def __init__(self, in_dim, out_dim, mu=23.359, std_dev=1.41):
        super().__init__()
        # used to normalise temperature from the model, obtain values from data:
        self.mu = mu
        self.std_dev = std_dev

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

    def get_action(self, x, unix_time):

        state = self.inputs_to_state(x, unix_time)

        prob_dist = torch.distributions.categorical.Categorical(logits=self.forward(state))  # make a probability distribution

        action = prob_dist.sample()  # sample from distribution pi(a|s) (action given state)

        return action, prob_dist.log_prob(action)

    def inputs_to_state(self, x, unix_time):
        """
        Convenience function to transform raw input from model into the NN state.

        x: torch.tensor,
            Tensor of temperatures from the model.

        unix_time: float,
            Unix epoch time in seconds.
        """

        if x.ndim == 0:
            x = x.unsqueeze(0)
        if unix_time.ndim == 0:
            unix_time = unix_time.unsqueeze(0)

        # normalise x using info obtained from data.
        x_norm = (x - self.mu) / self.std_dev
        # x_norm = torch.reshape(x_norm, (1,))

        day = 24 * 60 ** 2
        week = 7 * day
        # year = (365.2425) * day

        state = torch.stack([torch.flatten(x_norm),
                            np.sin(unix_time * (2 * np.pi / day)),
                            np.cos(unix_time * (2 * np.pi / day)),
                            np.sin(unix_time * (2 * np.pi / week)),
                            np.cos(unix_time * (2 * np.pi / week))]).to(torch.float32).T
        # state = state.to(torch.float32)

        # Adds broadcasting to function.
        # if len(unix_time) > 1:
        #     state = state.T

        # except TypeError:  # Occurs when no len()
        #     pass
        #
        # try:
        #     state = torch.cat((x_norm, state), dim=1)
        #
        # except RuntimeError:
        #     # Occurs when single time is being simulated
        #     state = torch.cat((x_norm, state.unsqueeze(0)), dim=1)

        return state

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

    def train(self, num_episodes, step_size):
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
                pred, reward = self.env.step(step_size)  # state and action produced in step

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
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, RC, time_data, temp_data):
        super().__init__()

        self.RC = RC  # RCModel Class
        self.time_data = time_data  # timeseries
        self.temp_data = temp_data
        self.t_index = 0  # used to keep track of index through timeseries
        self.loss_fn = torch.nn.MSELoss(reduction='none')

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

        # actions are decided and stored by the policy while integrating the ODE:
        pred = self.RC(t_eval)
        pred = pred.squeeze()

        actual = self.temp_data[self.t_index:int(self.t_index + step_size), 0:self.n_rooms]

        # negative so reward can be maximised
        reward = -self.loss_fn(pred[:, 2:], actual)  # Squared Error

        return pred.detach(), reward  # No need for grad on pred
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

