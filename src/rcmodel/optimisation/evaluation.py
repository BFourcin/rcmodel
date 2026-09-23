import numpy as np
import torch

from rcmodel.rc_model import LOAD_KEYS, PARAM_KEYS

"""
Note: The environment is likely to be wrapped.
We can still set environment variables using: env.unwrapped.variable = value
Then env.variable will still return value from the base environment.

However if we do env.variable = value, this will only set the variable in the wrapper.
"""


def make_update_env_fn(config):
    """Little function to enable interaction with the environment within Algorithm across all workers.
    >>> algo.env_runner_group.foreach_worker(make_update_env_fn(env_config))

    This is how a PBT exploit reaches the rollout workers: the mutated RC parameters go in
    `config` under "rc_parameters" and every worker's environment applies them (rebuilding
    A/B and re-solving iv_array) without the actor being torn down and rebuilt.
    """

    def update_env_conf(env):
        # update_from_config() does the parameter swap AND the RC.setup() that must follow
        # it, so there is deliberately no separate setup call here - doing both would
        # recompute iv_array over the whole dataset twice per exploit.
        env.unwrapped.update_from_config(config)

    def update_env_fn(worker):
        worker.foreach_env(update_env_conf)

    return update_env_fn


def evaluate(env, rl_algorithm, dataloader, record_window=None):
    """
    Score a policy + RC parameter set over a dataloader, one episode per batch.

    This is the metric PBT selects on, so it needs to be repeatable: score the same trial
    twice and get the same number. That requires a DETERMINISTIC dataloader - i.e. one over
    a BuildingTemperatureDataset, which walks consecutive windows, not a RandomSampleDataset,
    which draws a fresh random window on every access. Selecting on a noisy metric is how PBT
    ends up exploiting sampling luck instead of genuinely better parameters.

    The environment's own dataloader is restored before returning, so training carries on
    against the training data afterwards.

    Parameters
    ----------
    env : gym.Env
        The environment to evaluate in. May be wrapped.
    rl_algorithm : object
        Provides compute_single_action(obs, explore=False) -> action.
    dataloader : torch.utils.data.DataLoader
        Deterministic loader over the evaluation split.
    record_window : int or None
        Index of one window whose full trajectory should be recorded as it is scored - see
        the record's layout below. None (the default) records nothing. Recording reads what
        the scored episode already produced; it does not re-run anything.

    Returns
    -------
    reward_list : list of float
        One total episode reward per batch.
    record : dict or None
        For record_window, numpy arrays and scalars (all savable with np.savez):

        time            (T,)          seconds, every simulated row of the window
        states          (T, n_nodes)  predicted node temperatures: latent (wall) nodes
                                      first, then one column per room
        measured_time   (M,)          seconds, the window's data rows
        measured        (M, n_rooms)  measured room temperatures
        outdoor         (T,)          outdoor temperature at ``time``
        action          (S,)          action held over each step (1 = cooling on)
        action_start    (S,)          seconds, start of each step
        action_end      (S,)          seconds, end of each step
        step_reward     (S,)          reward of each step
        ghi             (T,)          global horizontal irradiance at ``time``, W/m2
        solar_w         (T,)          solar gain into all rooms at ``time``, W
        net_heat_w      (T,)          net heat input into all rooms at ``time``, W:
                                      gain + solar - cooling, with the action in force
        room_names, room_area, cool_w, gain_w   (n_rooms,)  per room; loads in W
        cool_w_m2, gain_w_m2                    (n_rooms,)  loads in W/m2
        solar_p                                 (n_rooms,)  fraction of GHI reaching each room
        param_names, param_values               the physical RC parameters
        window_index, window_reward             scalars
    """

    base_env = env.unwrapped  # Env is likely wrapped.
    original_dataloader = base_env.dataloader

    # Swapping the dataloader through the config also rebuilds iv_array for the new dataset.
    base_env.update_from_config({"dataloader": dataloader})

    reward_list = []
    record = None
    try:
        with torch.no_grad():
            for i in range(len(dataloader)):
                recording = i == record_window
                steps = []
                terminated = False
                truncated = False
                episode_reward = 0
                obs, _ = env.reset()
                while not (terminated or truncated):
                    # explore=False is what makes the score repeatable. The default
                    # (explore=None -> the algorithm's config, which is True) SAMPLES from
                    # the policy's action distribution, so scoring the same trial twice
                    # gives two different numbers and PBT selects on sampling luck.
                    action = rl_algorithm.compute_single_action(obs, explore=False)
                    step_start = base_env.observation[-1, 0].item()
                    obs, reward, terminated, truncated, _info = env.step(action)
                    episode_reward += reward
                    if recording:
                        # The base env's observation is the raw (unnormalised) trajectory of
                        # this step, without the row it shares with the previous step.
                        steps.append((int(action), step_start, reward, base_env.observation.clone()))

                reward_list.append(episode_reward)
                if recording:
                    record = _build_record(base_env, steps, i, episode_reward)
    finally:
        base_env.update_from_config({"dataloader": original_dataloader})

    return reward_list, record


def _build_record(base_env, steps, window_index, window_reward):
    """Assemble the record described in evaluate() from one recorded episode."""
    model = base_env.RC
    trajectory = torch.cat([observation for *_, observation in steps]).to(torch.float64)
    time = trajectory[:, 0].contiguous()  # the Tout interpolator warns on a strided column

    params, loads = model.get_physical_paramaters()
    rooms = model.building.rooms
    cool, gain, solar = (loads[LOAD_KEYS.index(key)] for key in ("cool", "gain", "solar"))

    # The action in force at every simulated row: each step's rows were simulated under its action.
    action_per_row = np.concatenate([np.full(len(observation), float(action)) for action, *_, observation in steps])
    heat_without_cooling = model._heat_input_watts(time, action=0).sum(axis=1)  # gain + solar, W
    cool_w = model.building.proportional_heating(cool).numpy().astype(np.float64)
    ghi = model.ghi(time)
    area = np.array([room.area for room in rooms], dtype=np.float64)

    return {
        "time": time.numpy(),
        "states": trajectory[:, 1:].numpy(),
        "measured_time": base_env.time_data.to(torch.float64).numpy(),
        "measured": base_env.temp_data[:, : base_env.n_rooms].numpy(),
        "outdoor": torch.as_tensor(model.Tout_continuous(time)).flatten().to(torch.float64).numpy(),
        "action": np.array([action for action, *_ in steps], dtype=np.int64),
        "action_start": np.array([start for _, start, *_ in steps], dtype=np.float64),
        "action_end": np.array([observation[-1, 0].item() for *_, observation in steps], dtype=np.float64),
        "step_reward": np.array([reward for _, _, reward, _ in steps], dtype=np.float64),
        "ghi": ghi,
        "solar_w": ghi * float(np.sum(solar.numpy().astype(np.float64) * area)),
        "net_heat_w": heat_without_cooling - action_per_row * cool_w.sum(),
        "room_names": np.array([room.name for room in rooms]),
        "room_area": np.array([room.area for room in rooms], dtype=np.float64),
        "cool_w_m2": cool.numpy().astype(np.float64),
        "gain_w_m2": gain.numpy().astype(np.float64),
        "solar_p": solar.numpy().astype(np.float64),
        "cool_w": cool_w,
        "gain_w": model.building.proportional_heating(gain).numpy().astype(np.float64),
        "param_names": np.array(PARAM_KEYS),
        "param_values": params.flatten().numpy().astype(np.float64),
        "window_index": window_index,
        "window_reward": float(window_reward),
    }
