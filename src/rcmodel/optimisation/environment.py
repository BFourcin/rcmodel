import pickle
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt

# Keys of LSIEnv's config that may be changed after construction. Everything else is
# structural (it changes the observation space or the episode shape) and needs a new env.
UPDATABLE_KEYS = ("dataloader", "rc_parameters", "update_state_dict")

# Changing either of these would change the observation space, so they are rejected outright.
FROZEN_KEYS = ("step_length", "render_mode")


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

    The reward is the negative MSE between the model's predicted room temperatures and
    the measured ones, i.e. this is a system-identification objective: a policy scores
    well by reproducing whatever cooling the real building actually did. The RC
    parameters are NOT learned here - they are fixed for the life of a PBT trial and
    supplied through the config (see update_from_config).

    ### Observation Space:
    [[unix_time, T_node1, T_node2, TRm1, TRm2, ...],    t0
    .                                                   t1
    .                                                   t2
    .                                                   tn
    ]
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array", "single_epoch_rgb_array"],
        "render_fps": 25,
    }

    def __init__(self, config: dict):
        super().__init__()

        self.config = dict(config)  # Own a copy: the caller's dict must not be mutated.
        self.config.setdefault("update_state_dict", None)
        self.config.setdefault("rc_parameters", None)
        self.RC = self.config["RC_model"]
        self.step_length = config["step_length"]
        self.render_mode = self.config.get("render_mode")
        self.dataloader = self.config["dataloader"]

        self.time_min = None  # used to help render graph. initialised in _init_render()
        self.time_max = None
        self.fig = None  # figure used in render
        # Persistent iterator over self.dataloader. reset() used to do next(iter(dataloader)),
        # which builds a FRESH iterator every episode and therefore always yields index 0.
        # RandomSampleDataset hid that (every index returns a different random window), but on
        # a deterministic dataset - which is what a stable evaluation metric needs - it means
        # every episode replays the same window. Held as state and advanced instead.
        self._batch_iter = None
        # init info dictionary:
        self.info = {}

        # get dt of data:
        t = self.dataloader.dataset[0][0]
        self.dt = int((t[1] - t[0]).item())

        self.day = 24 * 60**2

        self.step_size = int((self.step_length * 60) / self.dt)  # num rows of data needed for step_length minutes.
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
        self.action_space = spaces.Discrete(
            2,
        )

        # Observation is temperature of each room.
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.episode_info = {}  # for collecting render info.

        self.recording = bool(self.render_mode)

        # Build A/B and the initial-value array up front. The environment is constructed
        # from a config by every rollout worker, so it has to arrive ready to step - relying
        # on the caller to remember a separate setup() call is how a worker ends up
        # integrating from a stale (or absent) iv_array.
        self.update_from_config()
        if self.RC.A is None or self.RC.iv_array is None:
            self.RC.setup(self.dataloader.dataset)

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
        observation: np.array
            The observation at the end of the step.
        reward: float
            Negative MSE between predicted and measured room temperatures over the step.
        terminated: bool
            True if the batch is finished.
        truncated: bool
            Always False in this environment.
        info: dict
            A dictionary of information about the step.
        """
        with torch.no_grad():
            # solves an off by one issue caused by the iv technically being t0.
            # TODO: must be a more elegant way to do this.
            t_start = self.t_index - 1 if self.t_index > 0 else self.t_index

            t_end = int(self.t_index + self.step_size)

            # Take a sample of the time and temperature data
            t_eval = self.time_data[t_start:t_end]
            temperature_sample = self.temp_data[t_start:t_end, 0 : self.n_rooms]

            # set iv from last observation
            self.RC.iv = self.observation[-1, 1:].unsqueeze(0).T

            pred = self.RC(t_eval, action).squeeze()

            # negative so reward can be maximised.
            reward = -self.loss_fn(pred[:, 2:], temperature_sample)

            # remove first observation as this was the iv from the previous step
            # TODO: Tidy this up, there must be a better way.
            if self.t_index == 0:
                self.observation = torch.concat((t_eval.unsqueeze(0).T, pred.clone()), dim=1)
            else:
                self.observation = torch.concat((t_eval[1:].unsqueeze(0).T, pred[1:, :].clone()), dim=1)

            if self.render_mode is not None:
                self.episode_info["true_temperature"].extend(temperature_sample.numpy())
                self.episode_info["predicted_temperature"].extend(pred.numpy())
                self.episode_info["time"].extend(t_eval.unsqueeze(0).T.numpy())
                self.episode_info["actions"].extend([action, action])
                self.episode_info["t_actions"].extend([t_eval[0], t_eval[-1]])
                self.episode_info["reward"].append(reward.numpy())
                self.episode_info["t_reward"].append(t_eval[-1])

            # Check for done condition:
            if t_eval[-1] == self.time_data[-1]:
                self.terminated = True

            self.t_index += self.step_size
            self.step_count += 1  # Keep track of the number of steps taken.

            truncated = False  # Not used in this environment.

            return self.observation.numpy(), float(reward), self.terminated, truncated, self.info

    def reset(
        self,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
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
        self.episode_info["Q_watts"] = self.RC.building.proportional_heating(self.RC.cool_load)

        # get next batch from dataloader:
        self.time_data, self.temp_data = self._next_batch()

        self.time_data = self.time_data.squeeze(0)
        self.temp_data = self.temp_data.squeeze(0)

        # Find correct initial value for current start from pre-calculated array
        # if statement allows iv_array to be none and not cause reset() to fail.
        if self.RC.iv_array is not None:
            self.RC.iv = self.RC.iv_array(self.time_data[0])

        self.observation = self._get_obs()
        self.terminated = False

        self.need_init_render = True  # reset render logic
        plt.close("all")  # close any open figures

        return self.observation.numpy(), self.episode_info

    def _get_obs(self):
        return torch.concat((self.time_data[0].unsqueeze(0), self.RC.iv.flatten())).unsqueeze(0)

    def _next_batch(self):
        """Next batch from the dataloader, wrapping around at the end of an epoch.

        Note an InfiniteSampler dataloader never raises StopIteration, so the wrap-around
        only fires for a finite one (e.g. the deterministic dataset used for evaluation).
        """
        if self._batch_iter is None:
            self._batch_iter = iter(self.dataloader)
        try:
            return next(self._batch_iter)
        except StopIteration:
            self._batch_iter = iter(self.dataloader)
            return next(self._batch_iter)

    def __getstate__(self):
        # DataLoader iterators and matplotlib figures don't pickle, and Ray pickles
        # environments when it moves them between processes. Both are derived state that
        # rebuilds itself on demand, so drop them rather than trying to serialise them.
        state = self.__dict__.copy()
        state["_batch_iter"] = None
        state["fig"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._batch_iter = None
        self.fig = None

    def update_from_config(self, new_config=None):
        """
        Apply any pending updates from the config to the environment and model.

        This is the ONLY route by which a running environment's RC parameters change, and
        it is what a PBT exploit relies on: the new parameters arrive in the config (via
        make_update_env_fn across every rollout worker, or via a fresh env built from the
        trial's config) and land on the model here.

        Updatable keys are UPDATABLE_KEYS:
            dataloader        - swap the data the episodes are drawn from (e.g. train -> test).
            rc_parameters     - dict of 0-1 scaled RC parameters, see RCModel.set_parameters().
            update_state_dict - a full RCModel state_dict, an alternative to rc_parameters.

        Whenever any of these change, A/B are rebuilt and iv_array is recomputed for the
        current dataset, because both depend on the parameters.

        Anything in FROZEN_KEYS cannot be changed and raises. Unknown keys are ignored, so a
        caller can pass a whole trial config without filtering it first.
        """
        if new_config:
            for key in FROZEN_KEYS:
                if key in new_config and new_config[key] != self.config.get(key):
                    raise ValueError(f"Cannot change '{key}' on the fly - build a new environment instead.")

            for key in UPDATABLE_KEYS:
                if key in new_config:
                    self.config[key] = new_config[key]

        needs_setup = False

        dataloader = self.config.get("dataloader")
        if dataloader is not None and dataloader is not self.dataloader:
            self.dataloader = dataloader
            self._batch_iter = None  # the old iterator belongs to the old dataloader
            needs_setup = True

        # Both parameter routes are consumed (popped back to None) once applied, so a
        # later reset() doesn't redundantly re-apply and re-solve for iv_array.
        rc_parameters = self.config.get("rc_parameters")
        if rc_parameters:
            self.RC.set_parameters(rc_parameters)
            self.config["rc_parameters"] = None
            needs_setup = True

        state_dict = self.config.get("update_state_dict")
        if state_dict:
            self.RC.load_state_dict(state_dict)
            self.config["update_state_dict"] = None
            needs_setup = True

        if needs_setup:
            # Rebuilds A/B from the new parameters and re-solves the latent nodes' initial
            # values over the (possibly new) dataset.
            self.RC.setup(self.dataloader.dataset)

    # TODO: Better render.
    def render(self):
        if self.recording:
            if self.render_mode is not None:
                return self._render()
            return None
        else:
            # if not recording, return empty list
            return None

    def _render(self):
        assert self.render_mode in self.metadata["render_modes"]

        with torch.no_grad():
            # return empty list unless until episode is done.
            if self.render_mode in ["single_rgb_array", "single_epoch_rgb_array"] and not self.terminated:
                return None

            _line1, heat_line, ax, ax2 = self._init_render()

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

            ax.relim()
            ax.autoscale_view(tight=None, scalex=False, scaley=True)
            ax2.relim()
            ax2.autoscale_view(tight=None, scalex=False, scaley=True)
            self.fig.canvas.draw()

            if self.render_mode == "human":
                plt.pause(0.0001)
                return self.fig
            elif self.render_mode in {"rgb_array", "single_rgb_array", "single_epoch_rgb_array"}:
                # Return a numpy RGB array of the figure
                width, height = self.fig.get_size_inches() * self.fig.get_dpi()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype="uint8").reshape((int(height), int(width), 3))
                plt.close(self.fig)

                return img

    def _init_render(self):

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

            self.fig = plt.figure(figsize=(width, width * 0.75))

        if self.time_min is None:
            t, temp = self.dataloader.dataset.get_all_data()
            self.time_min = t[0]
            self.time_max = t[-1]
            self.time_all = t
            self.temp_data_all = temp[:, 0 : self.n_rooms]

        x = torch.arange(0, self.time_max - self.time_min, self.dt) / self.day
        y = torch.empty(len(x)) * torch.nan

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

        if self.RC.transform:
            gain = self.RC.scaling.physical_loads_scaling(self.RC.transform(self.RC.loads))[1, :]
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
        (heat_line,) = ax2.plot([0], [0], color="k", linestyle="--", alpha=0.5, label="heat ($W$)")

        lns = [line1, heat_line, gain_line, *ln2]
        labs = [line.get_label() for line in lns]
        ax.legend(lns, labs, loc="upper right")

        if self.render_mode == "human":
            self.fig.show()

        return line1, heat_line, ax, ax2

    def save_episode_info_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.episode_info, f)


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

    day = 24 * 60**2
    week = 7 * day

    state = [
        *x_norm.tolist(),
        np.sin(unix_time * (2 * np.pi / day)),
        np.cos(unix_time * (2 * np.pi / day)),
        np.sin(unix_time * (2 * np.pi / week)),
        np.cos(unix_time * (2 * np.pi / week)),
    ]

    return np.array(state)
