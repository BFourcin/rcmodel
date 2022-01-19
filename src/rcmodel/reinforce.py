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

from rcmodel.room import Room
from rcmodel.building import Building
from rcmodel.RCModel import RCModel
from rcmodel.tools import InputScaling
from rcmodel.tools import BuildingTemperatureDataset


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

        self.on_policy_reset()

    def forward(self, state):
        logits = self.linear_relu_stack(state)

        return logits

    def get_action(self, state):
        pd = torch.distributions.categorical.Categorical(logits=self.forward(state))  # make a probability distribution

        action = pd.sample()  # sample from distribution pi(a|s) (action given state)

        return action, pd.log_prob(action)

    def on_policy_reset(self):
        # this stores log_probs during an integration step.
        self.log_probs = []
        # self.rewards = []


class Reinforce:
    def __init__(self, env, time_data, temp_data, gamma=0.99, alpha=1e-3):

        assert len(time_data) == len(temp_data)

        self.env = env
        # self.pi = pi
        self.time_data = time_data
        self.temp_data = temp_data
        self.gamma = gamma
        self.alpha = alpha

        self.optimiser = torch.optim.Adam(self.env.RC.cooling_policy.parameters(), lr=self.alpha)

    def update_policy(self, rewards, log_probs):
        # downsample rewards by averaging each window.
        rewards = torch.tensor([window.mean() for window in torch.tensor_split(rewards, len(log_probs))])

        # Calculate Discounted Reward:
        discounted_rewards = torch.zeros(len(rewards))

        R = 0
        indx = len(rewards) - 1
        for r in reversed(rewards):
            R = r + self.gamma * R  # Discounted Reward is calculated from last reward to first.

            discounted_rewards[indx] = R  # Fill array back to front to un-reverse the order
            indx -= 1

        # Normalise rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)

        # discounted_rewards = torch.tensor(np.array(discounted_rewards.detach().numpy()))

        expected_reward = -torch.stack(log_probs) * discounted_rewards  # negative for maximising
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

        loss_fn = torch.nn.MSELoss(reduction='none')  # Squared Error

        # with tqdm(total=len(self.env.time_data) * num_episodes, position=0, leave=False) as pbar:  # progress bar
        for episode in range(num_episodes):
            self.env.reset()

            episode_rewards = []
            episode_ER = []

            # Time is increased in steps, with the policy updating after every step.
            while self.env.t_index < len(self.env.time_data) - 1:
                # takes a step_size forward in time
                pred = self.env.step(step_size).squeeze(-1)  # state and action produced in step

                actual = self.temp_data[self.env.t_index:int(self.env.t_index + step_size), 0:self.env.n_rooms]

                # negative so reward can be maximised
                reward = -loss_fn(pred[:, 2:], actual)

                # Do gradient decent on sample
                ER = self.update_policy(reward, self.env.RC.cooling_policy.log_probs)

                self.env.RC.cooling_policy.on_policy_reset()  # empty buffer

                # get last output and use for next initial value
                self.env.RC.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

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

    def __init__(self, RC, time_data):
        super().__init__()

        self.RC = RC  # RCModel Class
        self.time_data = time_data  # timeseries
        self.t_index = 0  # used to keep track of index through timeseries

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

        return pred.detach()  # No need for grad

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

if __name__ == '__main__':
    from main import initialise_model

    path_sorted = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/210813data_sorted.csv'
    time_data = torch.tensor(pd.read_csv(path_sorted, skiprows=0).iloc[:, 1], dtype=torch.float64)
    temp_data = torch.tensor(pd.read_csv(path_sorted, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
                             dtype=torch.float32)

    ######
    path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/'
    dt = 30  # timestep (seconds), data and the model are sampled at this frequency
    sample_size = int(5 * (60 ** 2 * 24) / dt)  # one day of data

    training_data = BuildingTemperatureDataset(path + 'train5d.csv', sample_size)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)
    ######




    time_data = time_data[0:100]
    temp_data = temp_data[0:100, :]

    policy = PolicyNetwork(7, 2)
    RC, Tout_continuous = initialise_model(policy)
    env = LSIEnv(RC, time_data)
    reinforce = Reinforce(env, time_data, temp_data, alpha=1e-2)

    num_episodes = 10
    step_size = 24*60**2 / 30  # timesteps in 1 day
    start_time = time.time()
    plot_total_rewards, plot_ER = reinforce.train(num_episodes, step_size)

    print(f'fin, duration: {(time.time() - start_time) / 60:.1f} minutes')

    fig, axs = plt.subplots(1, 2, figsize=(10, 7),)
    axs[0].plot(torch.stack(plot_ER).detach().numpy(), label='expected rewards')
    axs[0].legend()

    axs[1].plot(torch.stack(plot_total_rewards).detach().numpy(), label='total rewards')
    axs[1].legend()

    plt.savefig('Rewards.png')
    plt.show()

