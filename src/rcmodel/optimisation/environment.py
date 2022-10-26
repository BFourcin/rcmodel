import gym
from gym import spaces
from gym.utils.renderer import Renderer
from typing import Optional


import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
        logits = self.forward(state)

        # Debugging ----------------------
        if torch.isnan(logits).any():
            from datetime import datetime
            from pathlib import Path

            # datetime object containing current date and time
            now = datetime.now()
            # YY/mm/dd H:M:S
            dt_string = now.strftime("%y-%m-%d_%H:%M:%S")
            logfile = 'outputs/' + dt_string + '_errorlog' + '.csv'
            # Create dir if needed
            Path("./outputs/").mkdir(parents=True, exist_ok=True)

            s = state.detach().numpy()
            np.savetxt(logfile, s, delimiter=",")

            action = 0
            return action
        # ----------------------------------------------------------

        prob_dist = torch.distributions.categorical.Categorical(
            logits=logits)  # make a probability distribution

        action = prob_dist.sample()  # sample from distribution pi(a|s) (action given state)

        self.log_probs.append(prob_dist.log_prob(action))  # store log probability of action

        return action

    def on_policy_reset(self):
        # this stores log_probs during an integration step.
        self.log_probs = []
        # self.rewards = []


class LSIEnv(gym.Env):
    """Custom Environment that follows gym interface

    config = {"RC_model": rcmodel Class,
              "dataloader": torch.dataloader Object,
              "step_length": int (Minutes),
              "render_mode": str,
              }

    A dataloader is used to provide batches of time and temperature data to the model. The environment is run in
    steps of size step_size (15 mins default) with a fixed action, the steps will step through a batch of data until
    it is finished. This is one trajectory and we return the (observation, reward, done, info). At this point we
    expect self.reset() to be called to get the environment ready for the next step.

    The next step will then be a new trajectory and be from the next batch in the dataloader. Once all the batches have
    been seen we refresh the dataloader and go again from the start.

    ### Observation Space:
    [[unix_time, T_node1, T_node2, TRm1, TRm2, ...],    t0
    .                                                   t1
    .                                                   t2
    .                                                   tn
    ]
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 25,
    }

    def __init__(self, env_config: dict):
        super().__init__()

        self.RC = env_config["RC_model"]  # RCModel Class
        self.dataloader = env_config["dataloader"]
        # self.dataloader = self.RC.dataloader
        self.enum_data = None  # enumerate(dataloader)
        self.batch_idx = len(self.dataloader) - 1  # Keeps track of batch number in dataloader, initialised in reset()
        self.epochs = -1  # keeps count of the total epochs of data seen.
        self.time_min = None  # used to help render graph. initialised in _init_render()
        self.time_max = None
        self.fig = None  # figure used in render
        # init info dictionary:
        self.info = {"actions": [],  # place to record actions
                     "t_actions": [],  # record time action took place
                     }

        # get dt of data:
        t = self.dataloader.dataset[0][0]
        self.dt = int((t[1] - t[0]).item())

        self.day = 24 * 60 ** 2

        self.step_length = env_config["step_length"]  # Minutes
        self.step_size = int((self.step_length * 60) / self.dt)  # num rows of data needed for step_length minutes.
        self.t_index = 0  # used to keep track of index through timeseries
        self.loss_fn = torch.nn.MSELoss()

        # ----- GYM Stuff -----
        self.n_rooms = len(self.RC.building.rooms)

        time_low = [0]
        time_high = [np.float32(np.inf)]

        temp_low = [-np.float32(np.inf)] * (self.n_rooms + 2)  # +2 accounts for the latent nodes
        temp_high = [np.float32(np.inf)] * (self.n_rooms + 2)

        low = np.array([time_low + temp_low] * self.step_size)  # extend the vector by the number of timesteps
        high = np.array([time_high + temp_high] * self.step_size)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2, )

        # Observation is temperature of each room.
        self.observation_space = spaces.Box(low, high,
                                            dtype=np.float64)

        self.render_mode = env_config["render_mode"]
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.renderer = Renderer(self.render_mode, self._render)

    def step(self, action):
        # Execute a chunk of timeseries
        t_start = self.t_index - 1 if self.t_index > 0 else self.t_index  # solves an off by one issue caused by the iv technically being t0.
        t_eval = self.time_data[t_start:int(self.t_index + self.step_size)]
        temp = self.temp_data[t_start:int(self.t_index + self.step_size), 0:self.n_rooms]

        self.info["actions"].extend([action, action])  # record both start and end so we can plot actions later
        self.info["t_actions"].extend([t_eval[0], t_eval[-1]])

        # set iv from last observation
        self.RC.iv = self.observation[-1, 1:].unsqueeze(0).T

        pred = self.RC(t_eval, action)
        pred = pred.squeeze()

        # negative so reward can be maximised
        reward = -self.loss_fn(pred[:, 2:], temp).item()  # Mean Squared Error

        # remove first observation as this was the iv from the previous step
        if self.t_index == 0:
            self.observation = torch.concat((t_eval.unsqueeze(0).T, pred.detach().clone()), dim=1)
        else:
            self.observation = torch.concat((t_eval[1:].unsqueeze(0).T, pred[1:, :].detach().clone()), dim=1)

        if t_eval[-1] == self.time_data[-1]:  # if this condition is missed then error
            done = True
        else:
            done = False

        self.t_index += self.step_size

        self.renderer.render_step()
        return self.observation.numpy(), reward, done, self.info

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,
              ):
        super().reset(seed=seed)

        # Reset the state of the environment to an initial state
        self.t_index = 0

        self.info["actions"] = []  # reset recorded actions
        self.info["t_actions"] = []  # reset recorded action timeseries

        # check if we have reached end of data and need to reset enumerate.
        if self.batch_idx + 1 == len(self.dataloader):
            self.enum_data = enumerate(self.dataloader)
            self.epochs += 1

        # get next batch from dataloader:
        self.batch_idx, (self.time_data, self.temp_data) = next(self.enum_data)
        self.time_data = self.time_data.squeeze(0)
        self.temp_data = self.temp_data.squeeze(0)

        # Find correct initial value for current start from pre-calculated array
        self.RC.iv = self.RC.iv_array(self.time_data[0]).unsqueeze(0).T

        self.observation = torch.concat((self.time_data[0].unsqueeze(0), self.RC.iv.T.squeeze(0))).unsqueeze(0)

        self.need_init_render = True  # reset render logic
        self.renderer.reset()
        self.renderer.render_step()

        return self.observation.numpy()

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode='human'):
        # if self.render_mode:  # overwrite mode
        #     mode = self.render_mode

        assert mode in self.metadata["render_modes"]

        with torch.no_grad():
            # set up render environment
            if self.need_init_render:
                self._init_render(mode)
                self.need_init_render = False
                # img = None
                # if mode in {"rgb_array", "single_rgb_array"}:
                #     self.fig.canvas.draw()
                #     width, height = self.fig.get_size_inches() * self.fig.get_dpi()
                #     img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
                #
                # return img

            if mode == 'single_rgb_array':  # only do plot if it's the last timestep in episode:
                if not self.time_data[self.t_index+self.step_size-1] >= self.time_data[-1]:
                    return []

            # line1.set_data(self.observation[:, 0].numpy(), self.observation[:, 3:].numpy())
            ax.plot((self.observation[:, 0] - self.time_min).numpy() / self.day, self.observation[:, 3:].numpy(), 'k-')

            Q_watts = self.RC.building.proportional_heating(self.RC.cool_load)  # convert from W/m2 to W
            Q = self.info["actions"] * -Q_watts.detach().numpy()  # negative because cooling

            if len(Q) > 0:  # if not empty
                heat_line.set_data((torch.stack(self.info["t_actions"]) - self.time_min) / self.day, Q)

            # fig = plt.gcf()
            # ax = plt.gca()
            ax.relim()
            ax.autoscale_view(tight=None, scalex=False, scaley=True)
            ax2.relim()
            ax2.autoscale_view(tight=None, scalex=False, scaley=True)
            self.fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.draw()

            if mode == 'human':
                plt.pause(0.0001)
                return self.fig
            elif mode in {"rgb_array", "single_rgb_array"}:
                # Return a numpy RGB array of the figure
                width, height = self.fig.get_size_inches() * self.fig.get_dpi()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))

                return img

    def _init_render(self, mode):
        from matplotlib.lines import Line2D

        global line1, heat_line, ax, ax2

        if mode == 'human':
            plt.ion()
            self.fig = plt.gcf()
        else:
            plt.ioff()
            entrys = self.dataloader.dataset.entry_count
            inches_per_day = 0.5
            days = entrys/2880
            width = days*inches_per_day
            if width < 10:
                width = 10

            if self.fig is None:
                self.fig = plt.figure(figsize=(width, width*0.75))
            # canvas = FigureCanvas(fig)

        if self.time_min is None:
            t, temp = self.dataloader.dataset.get_all_data()
            self.time_min = t[0]
            self.time_max = t[-1]
            self.time_all = t
            self.temp_data_all = temp[:, 0:self.n_rooms]

        x = torch.arange(0, self.time_max - self.time_min, self.dt) / self.day
        y = torch.empty(len(x)) * torch.nan
        # y = torch.zeros(len(x))

        ax = self.fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.set_xlim([0, (self.time_max - self.time_min) / self.day])
        ax.set_title('Model Output')
        ax.set(xlabel='Time (days)', ylabel=r'Temperature ($^\circ$C)')
        ax2.set_ylabel(r"Heating/Cooling ($W$)")

        t_days_all = (self.time_all - self.time_min) / self.day
        line1, = ax.plot(x, y, 'k-', label=r'model ($^\circ$C)')
        ln2 = ax.plot(t_days_all.numpy(), self.temp_data_all[:, 0].numpy(), ':r', label=r'data ($^\circ$C)')
        # ln3 = ax.plot(t_days_all.numpy(), self.RC.Tout_continuous(self.time_all).numpy(), linestyle=':',
        #               color='darkorange', label=r'outside ($^\circ$C)')

        if self.RC.transform:
            gain = self.RC.scaling.physical_loads_scaling(self.RC.transform(self.RC.loads))[1, :]
        else:
            gain = self.RC.scaling.physical_loads_scaling(self.RC.loads)[1, :]

        gain_watts = gain * self.RC.building.rooms[0].area
        gain_line = ax2.axhline(gain_watts.detach().numpy(), linestyle='-.', color='k', alpha=0.5, label='gain ($W$)')

        # fake line so we can get a legend now. Real line is created in render()
        heat_line, = ax2.plot([0], [0], color='k', linestyle='--', alpha=0.5, label='heat ($W$)')

        lns = [line1, heat_line, gain_line] + ln2  # + ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right')

        if mode == 'human':
            self.fig.show()

        # return line1, heat_line, ax, ax2


class PreprocessEnv(gym.Wrapper):
    """
    Class used to preprocess the outputs from the environment before being seen by the policy.
    """

    def __init__(self, env, mu, std_dev):
        super().__init__(env)
        self.env = env
        self.mu = mu
        self.std_dev = std_dev

        time_high = [1.] * 4
        time_low = [-1.] * 4

        temp_high = [np.float32(np.inf)] * self.env.n_rooms  # This is normalised temperature so the limits are a guess.
        temp_low = [-np.float32(np.inf)] * self.env.n_rooms

        self.observation_space = spaces.Box(np.array(temp_low + time_low), np.array(temp_high + time_high),
                                            dtype=np.float64)

    def preprocess_observation(self, observation):
        """
        Function to transform observation to the state we want the policy to see/use.

        Used to wrap the original environment:
        env = gym.wrappers.TransformObservation(env, preprocess_observation)
        """

        # ndim will be different if from reset() or step()
        # if observation.ndim == 1:
        #     unix_time = observation[0]
        #     x = observation[3:]  # remove the latent nodes
        # else:
        #     unix_time = observation[-1, 0]
        #     x = observation[-1, 3:]  # remove the latent nodes

        unix_time = observation[-1, 0]
        x = observation[-1, 3:]  # remove the latent nodes

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

        return np.array(state)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.observation = self.preprocess_observation(observation)

        return self.observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.observation = self.preprocess_observation(observation)

        return self.observation


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

        pred = torch.zeros((1, 5))  # Fake pred to be compatible with Reinforce
        return pred, reward  # No need for grad on pred

    #          (observation, reward, done, info)
    # self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.t_index = 0
        self.RC.reset_iv()

        self.RC.cooling_policy.on_policy_reset()

    def render(self, mode='human'):
        # Render the environment to the screen

        return

# class Reinforce:
#     def __init__(self, env, gamma=0.99, alpha=1e-3):
#
#         self.env = env
#         # self.pi = pi
#         self.gamma = gamma
#         self.alpha = alpha
#
#         self.optimiser = torch.optim.Adam(self.env.RC.cooling_policy.parameters(), lr=self.alpha)
#
#     def update_policy(self, rewards, log_probs):
#         log_probs = torch.stack(log_probs).squeeze()  # get into right format regardless of whether single or multiple states have been used
#
#         # downsample rewards by averaging each window.
#         rewards = torch.tensor([window.mean() for window in torch.tensor_split(rewards, len(log_probs))])
#
#         # Calculate Discounted Reward:
#         discounted_rewards = torch.zeros(len(rewards))
#
#         R = 0
#         indx = len(rewards) - 1
#         for r in reversed(rewards):  # Future/Later rewards are discounted
#             R = r + self.gamma * R  # Discounted Reward is calculated from last reward to first.
#
#             discounted_rewards[indx] = R  # Fill array back to front to un-reverse the order
#             indx -= 1
#
#         # Normalise rewards
#         discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
#                 discounted_rewards.std() + 1e-9)
#
#         # discounted_rewards = torch.tensor(np.array(discounted_rewards.detach().numpy()))
#
#         expected_reward = -log_probs * discounted_rewards  # negative for maximising
#         expected_reward = torch.sum(expected_reward)
#         # print(f'ER {expected_reward}')
#
#         # Update parameters in pi
#         self.optimiser.zero_grad()
#         expected_reward.backward()
#         self.optimiser.step()
#
#         # print(list(self.pi.parameters())[0].grad)  # check on grads if needed
#
#         return expected_reward
#
#     def train(self, num_episodes):
#         self.env.RC.cooling_policy.train()  # Put in training mode
#         total_ER = []
#         total_rewards = []
#
#         # with tqdm(total=len(self.env.time_data) * num_episodes, position=0, leave=False) as pbar:  # progress bar
#         for episode in range(num_episodes):
#             self.env.reset()
#
#             episode_rewards = []
#             episode_ER = []
#
#             # Time is increased in steps, with the policy updating after every step.
#             while self.env.t_index < len(self.env.time_data) - 1:
#
#                 # takes a step_size forward in time
#                 pred, reward = self.env.step(action)  # state and action produced in step
#
#                 # Do gradient decent on sample
#                 ER = self.update_policy(reward, self.env.RC.cooling_policy.log_probs)
#
#                 self.env.RC.cooling_policy.on_policy_reset()  # empty buffer
#
#                 # get last output and use for next initial value
#                 # self.env.RC.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD
#
#                 episode_rewards.append(sum(reward))
#                 episode_ER.append(ER)
#
#                 self.env.t_index += int(step_size)  # increase environment time
#
#             # print(f'Episode {episode+1}, Expected Reward: {sum(episode_ER).item():.2f}, total_reward: {sum(episode_rewards).item():.2f}')
#
#             total_ER.append(sum(episode_ER).detach())
#             total_rewards.append(sum(episode_rewards).detach())
#
#         return total_rewards, total_ER
