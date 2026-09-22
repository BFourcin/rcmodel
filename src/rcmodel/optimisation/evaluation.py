import torch

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


def evaluate(env, rl_algorithm, dataloader):
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

    Returns
    -------
    reward_list : list of float
        One total episode reward per batch.
    render_list : list
        Rendered frames, empty unless the environment has a render_mode.
    """

    base_env = env.unwrapped  # Env is likely wrapped.
    original_dataloader = base_env.dataloader

    # Swapping the dataloader through the config also rebuilds iv_array for the new dataset.
    base_env.update_from_config({"dataloader": dataloader})

    reward_list = []
    render_list = []
    try:
        with torch.no_grad():
            for i in range(len(dataloader)):
                if base_env.render_mode == "single_epoch_rgb_array":
                    # Only render on the last episode.
                    base_env.recording = (i + 1) % len(dataloader) == 0
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
                    obs, reward, terminated, truncated, _info = env.step(action)
                    episode_reward += reward

                reward_list.append(episode_reward)
                render_list.append(env.render())
    finally:
        base_env.update_from_config({"dataloader": original_dataloader})

    return reward_list, remove_none(render_list)


def remove_none(nested_list):
    """
    Flatten a nested list and remove all occurrences of None values.

    Args:
        nested_list (list): The nested list to flatten and remove None values from.

    Returns:
        list: A flattened list with all None values removed.
    """
    flattened = []
    if isinstance(nested_list, list):
        for item in nested_list:
            if isinstance(item, list):
                flattened.extend(remove_none(item))
            elif item is not None:
                flattened.append(item)
    return flattened
