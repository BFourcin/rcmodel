import gymnasium as gym
from gymnasium import spaces
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
import pickle
import os
from collections import deque


# TODO: Remove POLICY NETWORK and all references to it.
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
            logfile = "outputs/" + dt_string + "_errorlog" + ".csv"
            # Create dir if needed
            Path("./outputs/").mkdir(parents=True, exist_ok=True)

            s = state.detach().numpy()
            np.savetxt(logfile, s, delimiter=",")

            action = 0
            return action
        # ----------------------------------------------------------

        prob_dist = torch.distributions.categorical.Categorical(
            logits=logits
        )  # make a probability distribution

        action = (
            prob_dist.sample()
        )  # sample from distribution pi(a|s) (action given state)

        self.log_probs.append(
            prob_dist.log_prob(action)
        )  # store log probability of action

        return action

    def on_policy_reset(self):
        # this stores log_probs during an integration step.
        self.log_probs = []
        # self.rewards = []


# TODO: Update gym to gymnasium
# TODO: Ensure that environment works with multiple batches.
class LSIEnv(gym.Env):
    """Custom Environment that follows gym interface

    config = {"RC_model": rcmodel Class,
              "dataloader": torch.dataloader Object,
              "step_length": int (Minutes),
              "render_mode": str,
              }

    A dataloader is used to provide batches of time and temperature data to the
    model. The environment is run in steps of size step_size (15 mins default) with a
    fixed action, the steps will step through a batch of data until it is finished.
    This is one trajectory and we return the (observation, reward, done, info). At
    this point we expect self.reset() to be called to get the environment ready for
    the next step.

    The next step will then be a new trajectory and be from the next batch in the
    dataloader. Once all the batches have been seen we refresh the dataloader and go
    again from the start.

    ### Observation Space:
    [[unix_time, T_node1, T_node2, TRm1, TRm2, ...],    t0
    .                                                   t1
    .                                                   t2
    .                                                   tn
    ]
    """

    metadata = {
        "render_modes": ["human",
                         "rgb_array",
                         "single_rgb_array",
                         "single_epoch_rgb_array"],
        "render_fps": 25,
    }

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.config["update_state_dict"] = config.get("update_state_dict", None)
        self.RC = config["RC_model"]
        self.step_length = config["step_length"]
        self.render_mode = config.get("render_mode", None)
        self._update_environment()  # Initialise dataloader and check for updates.
        # self.epochs_per_reset = env_config.get("epochs_per_reset", 1)

        # self.epochs = -1  # keeps count of the total epochs of data seen.
        self.time_min = None  # used to help render graph. initialised in _init_render()
        self.time_max = None
        self.fig = None  # figure used in render
        self.collect_rc_grad = False  # Flag to collect in rcModel or not.
        # init info dictionary:
        self.info = {}

        # get dt of data:
        t = self.dataloader.dataset[0][0]
        self.dt = int((t[1] - t[0]).item())

        self.day = 24 * 60 ** 2

        self.step_length = config["step_length"]  # Minutes
        self.step_size = int(
            (self.step_length * 60) / self.dt
        )  # num rows of data needed for step_length minutes.
        self.loss_fn = torch.nn.MSELoss()

        # ----- GYM Stuff -----
        self.n_rooms = len(self.RC.building.rooms)

        time_low = [0]
        time_high = [np.float32(np.inf)]

        temp_low = [-np.float32(np.inf)] * (
                self.n_rooms + 2
        )  # +2 accounts for the latent nodes
        temp_high = [np.float32(np.inf)] * (self.n_rooms + 2)

        low = np.array(
            [time_low + temp_low] * self.step_size
        )  # extend the vector by the number of timesteps
        high = np.array([time_high + temp_high] * self.step_size)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2,)

        # Observation is temperature of each room.
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.render_mode = config["render_mode"]
        assert (
                self.render_mode is None
                or self.render_mode in self.metadata["render_modes"]
        )
        self.episode_info = {}  # for collecting render info.

        if self.render_mode:
            self.recording = True
        else:
            self.recording = False

    def step(self, action):
        """
        The agent steps through the provided batch of data in steps of size step_size.
        The action is held constant for each step. For example a batch containing day of
        data is provided, step_size has been set to 15 mins. For each step the
        environment will integrate a 15min timeperiod and compare this with data to get
        the reward. Once all the data in the batch has been seen, 1 days worth in this
        example, the environment will report done=True and needs to be reset.

        Parameters
        ----------
        action: int
            0 or 1, 0 is off, 1 is on.

        Returns
        -------
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (float): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.


        observation: np.array
            The observation at the end of the step.
        reward: float
            The reward for the step.
        done: bool
            True if the batch is finished.
        info: dict
            A dictionary of information about the step.
        """
        with torch.set_grad_enabled(self.collect_rc_grad):
            # solves an off by one issue caused by the iv technically being t0.
            # TODO: must be a more elegant way to do this.
            if self.t_index > 0:
                t_start = self.t_index - 1
            else:
                t_start = self.t_index

            t_end = int(self.t_index + self.step_size)

            # Take a sample of the time and temperature data
            t_eval = self.time_data[t_start:t_end]
            temperature_sample = self.temp_data[t_start:t_end, 0:self.n_rooms]

            # record both start and end, so we can plot actions later
            # self.info["actions"].extend([action, action])
            # self.info["t_actions"].extend([t_eval[0], t_eval[-1]])

            # set iv from last observation
            self.RC.iv = self.observation[-1, 1:].unsqueeze(0).T

            pred = self.RC(t_eval, action).squeeze()

            # negative so reward can be maximised.
            reward = -self.loss_fn(pred[:, 2:], temperature_sample)

            # remove first observation as this was the iv from the previous step
            # TODO: Tidy this up, there must be a better way.
            if self.t_index == 0:
                self.observation = torch.concat(
                    (t_eval.unsqueeze(0).T, pred.detach().clone()), dim=1
                )
            else:
                self.observation = torch.concat(
                    (t_eval[1:].unsqueeze(0).T, pred[1:, :].detach().clone()), dim=1
                )

            if self.render_mode is not None:
                self.episode_info["true_temperature"].extend(temperature_sample.numpy())
                self.episode_info["predicted_temperature"].extend(pred.detach().numpy())
                self.episode_info["time"].extend(t_eval.unsqueeze(0).T.numpy())
                self.episode_info["actions"].extend([action, action])
                self.episode_info["t_actions"].extend([t_eval[0], t_eval[-1]])
                self.episode_info["reward"].append(reward.detach().numpy())
                self.episode_info["t_reward"].append(t_eval[-1])

            # Check for done condition:
            if t_eval[-1] == self.time_data[-1]:
                self.terminated = True

            self.t_index += self.step_size
            self.step_count += 1  # Keep track of the number of steps taken.

            if not self.collect_rc_grad:  # Don't need to save grads, keeps ray happy.
                reward = reward.detach().numpy()

            truncated = False  # Not used in this environment.

            return self.observation.numpy(), reward, self.terminated, truncated, self.info

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # Check to see if we need to update the models parameters, config could have
        # been updated externally.
        self.update_from_config()

        # Reset the state of the environment to an initial state
        self.t_index = 0
        self.step_count = 0

        self.episode_info["actions"] = deque()  # reset recorded actions
        self.episode_info["t_actions"] = deque()  # reset recorded action timeseries
        self.episode_info["true_temperature"] = deque()
        self.episode_info["predicted_temperature"] = deque()
        self.episode_info["time"] = deque()
        self.episode_info["reward"] = deque()
        self.episode_info["t_reward"] = deque()
        self.episode_info["Q_watts"] =\
            self.RC.building.proportional_heating(self.RC.cool_load)

        # get next batch from dataloader:
        self.time_data, self.temp_data = next(iter(self.dataloader))

        self.time_data = self.time_data.squeeze(0)
        self.temp_data = self.temp_data.squeeze(0)

        # Find correct initial value for current start from pre-calculated array
        # if statement allows iv_array to be none and not cause reset() to fail.
        if self.RC.iv_array:
            self.RC.iv = self.RC.iv_array(self.time_data[0])

        self.observation = self._get_obs()
        self.terminated = False

        self.need_init_render = True  # reset render logic
        plt.close("all")  # close any open figures

        return self.observation.numpy(), self.episode_info

    def _get_obs(self):
        return torch.concat(
            (self.time_data[0].unsqueeze(0), self.RC.iv.flatten())
        ).unsqueeze(0)

    def update_from_config(self, new_config=None):
        """
        Some environment parameters can be updated on the fly. This method checks for
        changes in updatable parameters between new_config and the current
        environment parameters and updates the environment accordingly.

        Parameters which can be updated on the fly are produced by:
        self._get_updatable_config()

        Only matching keys in the provided new_config are checked, everything else is
        ignored.
       """

        if new_config is None:
            new_config = self.config

        env_parameters = self._get_updatable_config()

        # Pop state_dict from new_config, we'll use it to update the model later.
        new_state_dict = new_config.pop('update_state_dict', None)
        env_parameters.pop('update_state_dict', None)  # Don't need anymore

        # Check if all keys in env_parameters are in new_config
        assert set(env_parameters.keys()).issubset(set(new_config.keys())), \
            'New config does not contain all keys of env_parameters.'

        # For all keys in env_parameters get differences between env_parameters and
        # new_config
        changed = set(env_parameters.items()).difference(set(new_config.items()))

        for key, _ in iter(changed):
            assert key not in ("step_length", "render_mode"), \
                "Cannot change step_length or render_mode on the fly."

            self.config[key] = new_config[key]

        if changed:
            self._update_environment()

        if new_state_dict:
            self.RC.load_state_dict(new_state_dict)
            self.config["update_state_dict"] = None

    # TODO: Better render.
    def render(self):
        if self.recording:
            if self.render_mode is not None:
                return self._render()
        else:
            # if not recording, return empty list
            return None

    def _render(self):
        # if self.render_mode:  # overwrite mode
        #     mode = self.render_mode

        assert self.render_mode in self.metadata["render_modes"]

        with torch.no_grad():
            # set up render environment
            # if self.need_init_render:
            #     self._init_render(mode)
            #     self.need_init_render = False

            # return empty list unless until episode is done.
            if self.render_mode in ["single_rgb_array", "single_epoch_rgb_array"]:
                if not self.terminated:
                    return None

            line1, heat_line, ax, ax2 = self._init_render()

            # line1.set_data(self.observation[:, 0].numpy(), self.observation[:, 3:].numpy())

            # Plot the predicted temperature
            y = np.array(self.episode_info["predicted_temperature"])
            x = np.array(self.episode_info["time"])
            ax.plot((x[:, 0] - self.time_min.numpy()) / self.day, y[:, 2:], "k-")

            # convert from W/m2 to W
            Q_watts = self.RC.building.proportional_heating(self.RC.cool_load)
            # negative because cooling:
            Q = np.array(self.episode_info["actions"]) * -Q_watts.detach().numpy()

            if len(Q) > 0:  # if not empty
                # plot cooling line in Watts
                t = np.array(self.episode_info["t_actions"]) - self.time_min.numpy()
                heat_line.set_data(t / self.day, Q)

            # fig = plt.gcf()
            # ax = plt.gca()
            ax.relim()
            ax.autoscale_view(tight=None, scalex=False, scaley=True)
            ax2.relim()
            ax2.autoscale_view(tight=None, scalex=False, scaley=True)
            self.fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.draw()

            if self.render_mode == "human":
                plt.pause(0.0001)
                return self.fig
            elif self.render_mode in {"rgb_array",
                                      "single_rgb_array",
                                      "single_epoch_rgb_array"}:
                # Return a numpy RGB array of the figure
                width, height = self.fig.get_size_inches() * self.fig.get_dpi()
                img = np.frombuffer(
                    self.fig.canvas.tostring_rgb(), dtype="uint8"
                ).reshape((int(height), int(width), 3))
                plt.close(self.fig)

                return img

    def _init_render(self):
        from matplotlib.lines import Line2D

        # global line1, heat_line, ax, ax2

        if self.render_mode == "human":
            plt.ion()
            self.fig = plt.gcf()
        else:
            plt.ioff()
            entrys = self.dataloader.dataset.entry_count
            inches_per_day = 0.5
            days = entrys / 2880
            width = days * inches_per_day
            if width < 10:
                width = 10

            # if self.fig is None:
            #     self.fig = plt.figure(figsize=(width, width * 0.75))
            # canvas = FigureCanvas(fig)
            self.fig = plt.figure(figsize=(width, width * 0.75))

        if self.time_min is None:
            t, temp = self.dataloader.dataset.get_all_data()
            self.time_min = t[0]
            self.time_max = t[-1]
            self.time_all = t
            self.temp_data_all = temp[:, 0: self.n_rooms]

        x = torch.arange(0, self.time_max - self.time_min, self.dt) / self.day
        y = torch.empty(len(x)) * torch.nan
        # y = torch.zeros(len(x))

        ax = self.fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.set_xlim([0, (self.time_max - self.time_min) / self.day])
        ax.set_title("Model Output")
        ax.set(xlabel="Time (days)", ylabel=r"Temperature ($^\circ$C)")
        ax2.set_ylabel(r"Heating/Cooling ($W$)")

        t_days_all = (self.time_all - self.time_min) / self.day
        (line1,) = ax.plot(x, y, "k-", label=r"model ($^\circ$C)")
        ln2 = ax.plot(
            t_days_all.numpy(),
            self.temp_data_all[:, 0].numpy(),
            ":r",
            label=r"data ($^\circ$C)",
        )
        # ln3 = ax.plot(t_days_all.numpy(), self.RC.Tout_continuous(self.time_all).numpy(), linestyle=':',
        #               color='darkorange', label=r'outside ($^\circ$C)')

        if self.RC.transform:
            gain = self.RC.scaling.physical_loads_scaling(
                self.RC.transform(self.RC.loads)
            )[1, :]
        else:
            gain = self.RC.scaling.physical_loads_scaling(self.RC.loads)[1, :]

        gain_watts = gain * self.RC.building.rooms[0].area
        gain_line = ax2.axhline(
            gain_watts.detach().numpy(),
            linestyle="-.",
            color="k",
            alpha=0.5,
            label="gain ($W$)",
        )

        # fake line so we can get a legend now. Real line is created in render()
        (heat_line,) = ax2.plot(
            [0], [0], color="k", linestyle="--", alpha=0.5, label="heat ($W$)"
        )

        lns = [line1, heat_line, gain_line] + ln2  # + ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="upper right")

        if self.render_mode == "human":
            self.fig.show()

        return line1, heat_line, ax, ax2

    def save_episode_info_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.episode_info, f)

    def _get_updatable_config(self):
        config = {
            # "RC_model": self.RC,
            "dataloader": self.dataloader,
            # "step_length": self.step_length,
            # "render_mode": self.render_mode,
            "update_state_dict": None,
        }
        return config

    def _update_environment(self):
        """Sets the environment to the new config."""

        # self.RC = self.config["RC_model"]
        self.dataloader = self.config["dataloader"]
        # self.batch_generator = iter(self.dataloader) # this was causing pickle issues
        # just remove it.
        # self.step_length = self.config["step_length"]
        # self.render_mode = self.config["render_mode"]


class PreprocessEnv(gym.ObservationWrapper):
    """
    A gym observation wrapper that preprocesses the observations.

    This wrapper applies a normalization transformation to the observations by
    subtracting the mean and dividing by the standard deviation.

    Also sin(time) and cos(time) are added to the observations.

       Args:
           env (gym.Env): The environment to wrap and preprocess observations for.
           mu (float): Mean of the data used for normalization.
           std_dev (float): Standard deviation of the data used for normalization.

       Attributes:
           mu (float): Mean of the data used for normalization.
           std_dev (float): Standard deviation of the data used for normalization.
           observation_space (gym.spaces.Box): The modified observation space after
           preprocessing.
    """
    def __init__(self, env, mu, std_dev):
        super().__init__(env)

        self.mu = mu
        self.std_dev = std_dev

        time_high = [1.0] * 4
        time_low = [-1.0] * 4

        # This is normalised temperature so the limits are a guess.
        temp_high = [np.float32(np.inf)] * env.n_rooms
        temp_low = [-np.float32(np.inf)] * env.n_rooms

        self.observation_space = spaces.Box(
            np.array(temp_low + time_low),
            np.array(temp_high + time_high),
            dtype=np.float64,
        )

    def observation(self, observation):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        unix_time = observation[-1, 0]
        x = observation[-1, 3:]  # remove the latent nodes

        return preprocess_observation(x, unix_time, self.mu, self.std_dev)


def preprocess_observation(x, unix_time, mu, std_dev):
    """
    Function to transform observation to the state we want the policy to see/use.

    Used to wrap the original environment:
    env = gym.wrappers.TransformObservation(env, preprocess_observation)

    Parameters
        ----------
        x: torch.tensor
            tensor of non latent nodes, i.e. room temperatures.
        unix_time : float
            Time at observation.
        mu: float
            Mean of data.
        std_dev: float
            Standard deviation of data.
    """

    # normalise x using info obtained from data.
    x_norm = (x - mu) / std_dev

    day = 24 * 60 ** 2
    week = 7 * day
    # year = (365.2425) * day

    state = x_norm.tolist() + [
        np.sin(unix_time * (2 * np.pi / day)),
        np.cos(unix_time * (2 * np.pi / day)),
        np.sin(unix_time * (2 * np.pi / week)),
        np.cos(unix_time * (2 * np.pi / week)),
    ]

    return np.array(state)
