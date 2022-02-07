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

        # normalise x using info obtained from data.
        x_norm = (x - self.mu) / self.std_dev
        # x_norm = torch.reshape(x_norm, (1,))

        day = 24 * 60 ** 2
        week = 7 * day
        # year = (365.2425) * day

        time_signals = torch.stack([np.sin(unix_time * (2 * np.pi / day)),
                                    np.cos(unix_time * (2 * np.pi / day)),
                                    np.sin(unix_time * (2 * np.pi / week)),
                                    np.cos(unix_time * (2 * np.pi / week))])

        try:  # Adds broadcasting to function.
            if len(unix_time) > 1:
                time_signals = time_signals.T

        except TypeError:  # Occurs when no len()
            pass

        state = torch.cat((x_norm, time_signals), dim=1)

        return state

    def on_policy_reset(self):
        # this stores log_probs during an integration step.
        self.log_probs = []
        # self.rewards = []


class Reinforce:
    def __init__(self, env, gamma=0.99, alpha=1e-3):

        assert len(time_data) == len(temp_data)

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
        for r in reversed(rewards):
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


        # for i in range(len(dummy_temp)):
        #     policy_action, log_prob = self.RC.cooling_policy.get_action(dummy_temp[i], t_eval[i])
        #
        #     if self.RC.cooling_policy.training:  # if in training mode store log_prob
        #         self.RC.cooling_policy.log_probs.append(log_prob)
        #
        #     policy_actions.append(policy_action)

        prior_actions = self.prior_data[self.t_index:int(self.t_index + step_size), 0:self.n_rooms]  # read in
        policy_actions = torch.tensor(policy_action).unsqueeze(0).T

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





if __name__ == '__main__':
    # Remember to pip install latest package as following import uses compiled package not relative imports
    from rcmodel import initialise_model, PriorCoolingPolicy, InputScaling

    import matplotlib.pyplot as plt
    from pathlib import Path

    path_sorted = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/210813data_sorted.csv'
    # time_data = torch.tensor(pd.read_csv(path_sorted, skiprows=0).iloc[:, 1], dtype=torch.float64)
    # temp_data = torch.tensor(pd.read_csv(path_sorted, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
    #                          dtype=torch.float32)

    actions = []
    hot_rm = []

    day = 24*60**2

    pi = PolicyNetwork(5, 2)
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    Q_limit = [-100, 100]  # Cooling limit and gain limit in W/m2
    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather ' \
                        'Files/JuneSept.csv'
    model = initialise_model(pi, scaling, weather_data_path)

    time_data = torch.arange(0, 30 * day, 30)
    temp_data = 3.5*torch.sin(time_data * (2 * np.pi/day) - 2/3*day) + 22.5  # dummy rm temperature data.
    temp_data = temp_data.unsqueeze(0).T
    env = PriorEnv(model, time_data, temp_data, PriorCoolingPolicy())

    rl = Reinforce(env)

    # rl.train(1, day)

    rewards_plot = []
    ER_plot = []
    opt_id = 0
    Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

    epochs = 400

    for epoch in trange(epochs):

        # Policy Training:
        num_episodes = 1
        step_size = day
        rewards, ER = rl.train(num_episodes, step_size)
        rewards_plot.append(rewards)
        ER_plot.append(ER)

        tqdm.write(f'Epoch {epoch + 1}, Policy Rewards {sum(rewards).item():.1f}, Policy Expected Rewards {sum(ER).item():.1f}')

    fig, axs = plt.subplots(1, 2, figsize=(10, 7), dpi=400)
    axs[0].plot(range(1, epochs + 1), torch.flatten(torch.tensor(ER_plot)).detach().numpy(), 'b',
                label='expected rewards')
    axs[0].legend()

    axs[1].plot(range(1, epochs + 1), torch.flatten(torch.tensor(rewards_plot)).detach().numpy(), 'r',
                label='total rewards')
    axs[1].legend()
    fig.suptitle('Rewards', fontsize=16)
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs[0].set_ylabel('Reward')
    axs[1].set_ylabel('Reward')

    plt.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
    plt.show()
    # plt.close()




