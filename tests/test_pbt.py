"""
Tests for the PBT search over RC parameters.

The invariant under test throughout is the one the whole design rests on:

    RC parameters come from the trial's config, and ONLY from the config.
    The checkpoint carries policy weights and nothing else.

A PBT exploit copies a good trial's checkpoint into a bad trial and mutates the bad trial's
config. If parameters could also arrive from the checkpoint, the mutation would be silently
overwritten and the search would go nowhere while still looking like it was working.
"""

import copy

import numpy as np
import pytest
import ray
import torch

from rcmodel import (
    LOAD_KEYS,
    PARAM_KEYS,
    RC_PARAM_KEYS,
    env_creator,
    evaluate,
    load_model_record,
    make_dataloaders,
    model_creator,
)
from rcmodel.optimisation.pbt import (
    IMPLAUSIBLE_PENALTY,
    METRIC,
    RCPolicyTrainable,
    build_tuner,
    physical_to_scaled,
    sample_plausible_population,
    search_space,
    slowest_time_constant_days,
)

torch.set_num_threads(1)


# --------------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def ray_cluster():
    """A small local Ray cluster shared by the tests that need one.

    Capped at 2 CPUs so the suite stays polite on a laptop, which is where this is expected
    to run first.
    """
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    yield
    ray.shutdown()


def tiny_ppo_settings():
    """PPO settings small enough that an iteration is a few seconds.

    num_env_runners=0 keeps the rollouts in-process, so an environment is created on the
    local worker and the tests can inspect it directly.
    """
    return {
        "num_env_runners": 0,
        "train_batch_size": 32,
        "minibatch_size": 16,
        "rollout_fragment_length": 32,
    }


def trial_config(model_config, env_config, physical, **overrides):
    """A complete RCPolicyTrainable config for one set of physical parameters."""
    config = {
        "model_config": model_config,
        "env_config": {**env_config, "model_config": model_config},
        "ppo": tiny_ppo_settings(),
        "max_time_constant_days": None,
        "evaluation_interval": 1,
    }
    config.update(physical)
    config.update(overrides)
    return config


def env_parameters(algo):
    """The RC parameters actually held by every environment behind an algorithm.

    Reads through to the environments the policy is really training in, rather than
    trusting the trainable's own bookkeeping - the point of these tests is that the two
    agree.
    """

    def read(worker):
        return worker.foreach_env(lambda env: env.unwrapped.RC.get_parameters())

    per_worker = algo.env_runner_group.foreach_worker(read)
    return [params for worker_result in per_worker for params in (worker_result or [])]


def assert_parameters_close(actual, expected):
    for key in RC_PARAM_KEYS:
        np.testing.assert_allclose(
            np.asarray(actual[key], dtype=float),
            np.asarray(expected[key], dtype=float),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"parameter '{key}' does not match",
        )


# --------------------------------------------------------------------------- scaling


def _physical_values(model):
    """{name: physical value(s)} the model is actually running."""
    params, loads = model.get_physical_paramaters()
    values = {key: params.flatten()[i].item() for i, key in enumerate(PARAM_KEYS)}
    values.update({key: loads[i].numpy() for i, key in enumerate(LOAD_KEYS)})
    return values


def test_out_of_range_values_pass_through_unclipped(get_model_config):
    """Values outside the configured range are run as given, not clipped.

    PBT perturbs a parameter by multiplying it (0.8x / 1.2x by default) without re-clipping,
    so a trial's config walks out of its range. The model must run exactly what the config
    says - if anything clips on the way in, the logged config and the model disagree, and two
    trials reporting different values silently run the same model.
    """
    above = {key: get_model_config[key][1] * 1.5 for key in RC_PARAM_KEYS}
    below = {key: get_model_config[key][0] * 0.5 for key in PARAM_KEYS}
    # The loads' configured floor is 0 - there is no physically valid value below it.
    below.update({key: sum(get_model_config[key]) / 2 for key in LOAD_KEYS})

    scaled_above = physical_to_scaled(get_model_config, above)
    scaled_below = physical_to_scaled(get_model_config, below)
    # Guard against the test passing vacuously: these really are outside machine space's [0, 1].
    assert all(scaled_above[key] > 1 for key in RC_PARAM_KEYS)
    assert all(scaled_below[key] < 0 for key in PARAM_KEYS)

    for physical, scaled in ((above, scaled_above), (below, scaled_below)):
        model = model_creator({**get_model_config, "parameters": scaled})
        running = _physical_values(model)
        for key in RC_PARAM_KEYS:
            np.testing.assert_allclose(running[key], physical[key], rtol=1e-5, err_msg=f"'{key}' was not run as given")


@pytest.mark.parametrize(
    "key, value",
    [("Rin", 0.0), ("R1", -0.1), ("C1", 0.0), ("C_rm", -5.0), ("cool", -1.0), ("gain", -0.5)],
)
def test_physically_invalid_parameters_are_rejected(get_model_config, physical_params, key, value):
    """Out of range is allowed; physically impossible is not.

    A must be built from 1/(R*C), so resistances and capacitances have to be strictly
    positive, and a negative cooling or gain limit would reverse the direction of the load.
    """
    physical = dict(physical_params[0])
    physical[key] = value
    with pytest.raises(ValueError, match=key):
        model_creator({**get_model_config, "parameters": physical_to_scaled(get_model_config, physical)})


def test_zero_loads_are_physically_valid(get_model_config, physical_params):
    """No cooling and no gain are meaningful buildings, not errors."""
    physical = dict(physical_params[0])
    physical["cool"] = 0.0
    physical["gain"] = 0.0
    model = model_creator({**get_model_config, "parameters": physical_to_scaled(get_model_config, physical)})
    running = _physical_values(model)
    assert np.all(running["cool"] == 0.0)
    assert np.all(running["gain"] == 0.0)


def test_physical_to_scaled_round_trips(get_model_config, physical_params):
    """A mid-range physical value must scale to the fraction of the range it sits at."""
    quarter, _ = physical_params
    scaled = physical_to_scaled(get_model_config, quarter)
    for key in RC_PARAM_KEYS:
        np.testing.assert_allclose(np.asarray(scaled[key]), 0.25, rtol=1e-6)


# --------------------------------------------------------------------------- initial population


def restrictive_threshold(model_config, quantile=0.25, n=20, seed=0):
    """A time-constant threshold most prior draws fail, so rejection genuinely happens.

    Taken from the fixture building's own distribution rather than hardcoded, because the
    time constants depend on the building geometry as much as on the ranges.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        space = search_space(model_config)
        taus = [
            slowest_time_constant_days(model_config, {key: float(space[key].sample()) for key in RC_PARAM_KEYS})
            for _ in range(n)
        ]
    finally:
        np.random.set_state(rng_state)
    return float(np.quantile(taus, quantile))


def test_sample_plausible_population_respects_threshold(get_model_config):
    """Every member of the initial population passes the plausibility filter."""
    threshold = restrictive_threshold(get_model_config)
    population = sample_plausible_population(get_model_config, 4, threshold)

    assert len(population) == 4
    for draw in population:
        assert set(draw) == set(RC_PARAM_KEYS)
        assert slowest_time_constant_days(get_model_config, draw) <= threshold
    assert len({tuple(sorted(draw.items())) for draw in population}) == 4, "draws should be independent"


def test_sample_plausible_population_raises_when_unreachable(get_model_config):
    """A threshold nothing can meet fails loudly instead of looping forever."""
    with pytest.raises(RuntimeError, match="max_time_constant_days"):
        sample_plausible_population(get_model_config, 2, 1e-12, max_draws=10)


def test_capacitances_are_sampled_log_uniformly(get_model_config):
    """Capacitance ranges span orders of magnitude, so uniform sampling would waste the search.

    C1 covers 1e5 to 1e8: sampled uniformly, ~99% of draws land in the top decade and a
    small-capacitance building would never be tried. Sampled log-uniformly, the decades are
    drawn about evenly - which is what this asserts, via the median landing near the
    geometric rather than the arithmetic mean.
    """
    space = search_space(get_model_config, log_uniform=True)
    draws = np.array([space["C1"].sample() for _ in range(2000)])

    low, high = get_model_config["C1"]
    geometric_mean = np.sqrt(low * high)
    arithmetic_mean = (low + high) / 2

    median = np.median(draws)
    assert abs(np.log10(median) - np.log10(geometric_mean)) < 0.2
    assert median < arithmetic_mean / 10, "draws look uniform, not log-uniform"


def test_loads_are_sampled_uniformly(get_model_config):
    """The load ranges start at 0, where log-uniform is undefined - and zero cooling is a
    physically meaningful draw the search should be able to make."""
    space = search_space(get_model_config, log_uniform=True)
    draws = np.array([space["cool"].sample() for _ in range(2000)])

    low, high = get_model_config["cool"]
    assert abs(np.median(draws) - (low + high) / 2) < 0.1 * (high - low)


# --------------------------------------------------------------------------- environment


def test_env_creator_does_not_mutate_its_config(get_model_config, env_config):
    """env_creator used to write the constructed model back into env_config under "RC_model".

    Tune deep-copies a trial's config, so a live model object sitting in it means a trial's
    parameters stop being described by its config - and a PBT exploit, which only changes
    the config, would silently do nothing.
    """
    config = {**env_config, "model_config": get_model_config}
    before = copy.deepcopy({key: value for key, value in config.items() if key != "model_config"})

    env_creator(config)

    assert "RC_model" not in config, "env_creator must not put a live model in the config"
    after = {key: value for key, value in config.items() if key != "model_config"}
    assert set(after) == set(before)


def test_env_parameters_come_from_the_config(get_model_config, env_config, physical_params):
    """Two environments built from configs differing only in their parameters must differ."""
    low_set, high_set = physical_params

    def build(physical):
        model_config = copy.deepcopy(get_model_config)
        model_config["parameters"] = physical_to_scaled(get_model_config, physical)
        return env_creator({**env_config, "model_config": model_config})

    env_low = build(low_set)
    env_high = build(high_set)

    assert_parameters_close(env_low.unwrapped.RC.get_parameters(), physical_to_scaled(get_model_config, low_set))
    assert_parameters_close(env_high.unwrapped.RC.get_parameters(), physical_to_scaled(get_model_config, high_set))
    assert not torch.allclose(env_low.unwrapped.RC.A, env_high.unwrapped.RC.A), (
        "different parameters produced the same system matrix"
    )


def test_env_parameter_update_in_place(get_model_config, env_config, physical_params):
    """update_from_config must swap parameters AND rebuild everything derived from them.

    This is the path a PBT exploit takes into a running rollout worker (see
    make_update_env_fn). If A/B or iv_array were left stale, the policy would keep training
    against the old building while the trial reported the new parameters.
    """
    low_set, high_set = physical_params

    model_config = copy.deepcopy(get_model_config)
    model_config["parameters"] = physical_to_scaled(get_model_config, low_set)
    env = env_creator({**env_config, "model_config": model_config})

    matrix_before = env.unwrapped.RC.A.clone()
    iv_before = env.unwrapped.RC.iv_array(env.unwrapped.dataloader.dataset.get_all_data()[0][0]).clone()

    env.unwrapped.update_from_config({"rc_parameters": physical_to_scaled(get_model_config, high_set)})

    assert_parameters_close(env.unwrapped.RC.get_parameters(), physical_to_scaled(get_model_config, high_set))
    assert not torch.allclose(env.unwrapped.RC.A, matrix_before), "A was not rebuilt from the new parameters"

    iv_after = env.unwrapped.RC.iv_array(env.unwrapped.dataloader.dataset.get_all_data()[0][0]).clone()
    assert not torch.allclose(iv_after, iv_before), "iv_array was not recomputed for the new parameters"


def test_env_rejects_structural_changes(get_model_config, env_config):
    """step_length changes the observation space, so it cannot be swapped into a running
    environment - better a clear error than a silently mis-shaped policy."""
    config = {**env_config, "model_config": get_model_config}
    env = env_creator(config)

    with pytest.raises(ValueError, match="step_length"):
        env.unwrapped.update_from_config({"step_length": 60})


def test_episodes_walk_distinct_windows(get_model_config, data_config, env_config):
    """reset() must advance through the dataloader rather than restarting it.

    It used to call next(iter(dataloader)), which builds a fresh iterator every episode and
    therefore always yields index 0. RandomSampleDataset hid that (every access returns a
    different random window), but the deterministic evaluation dataset does not - every
    evaluation episode would have replayed the same window.
    """
    _, eval_dataloader = make_dataloaders(data_config)
    if len(eval_dataloader) < 2:
        pytest.skip("evaluation split is only one window long; nothing to advance through")

    model_config = copy.deepcopy(get_model_config)
    env = env_creator({**env_config, "model_config": model_config, "dataloader": eval_dataloader})

    env.reset()
    first_window = env.unwrapped.time_data.clone()
    env.reset()
    second_window = env.unwrapped.time_data.clone()

    assert not torch.equal(first_window, second_window)


# --------------------------------------------------------------------------- metric


def test_evaluate_is_deterministic(get_model_config, data_config, env_config, ray_cluster):
    """Scoring the same trial twice must give the same number.

    PBT compares trials on this metric, so noise in it gets selected on as if it were
    signal - a trial that got a lucky window would be cloned across the population. A
    deterministic evaluation split is what rules that out.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    register_env("LSIEnv", env_creator)
    _, eval_dataloader = make_dataloaders(data_config)

    model_config = copy.deepcopy(get_model_config)
    env = env_creator({**env_config, "model_config": model_config})

    algo = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="LSIEnv", env_config={**env_config, "model_config": model_config}, disable_env_checking=True)
        .env_runners(num_env_runners=0, rollout_fragment_length=32)
        .training(train_batch_size=32, minibatch_size=16)
        .framework("torch")
        .build()
    )
    try:
        first, _ = evaluate(env, algo, eval_dataloader)
        second, _ = evaluate(env, algo, eval_dataloader)
    finally:
        algo.stop()

    assert first, "evaluation produced no episodes"
    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


def test_evaluate_restores_the_training_dataloader(get_model_config, data_config, env_config, ray_cluster):
    """Evaluation borrows the environment, so it must hand it back.

    Otherwise training silently continues against the test split after the first
    evaluation - which both corrupts the metric and leaks the test set into training.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    register_env("LSIEnv", env_creator)
    _, eval_dataloader = make_dataloaders(data_config)

    model_config = copy.deepcopy(get_model_config)
    env = env_creator({**env_config, "model_config": model_config})
    training_dataloader = env.unwrapped.dataloader

    algo = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="LSIEnv", env_config={**env_config, "model_config": model_config}, disable_env_checking=True)
        .env_runners(num_env_runners=0, rollout_fragment_length=32)
        .training(train_batch_size=32, minibatch_size=16)
        .framework("torch")
        .build()
    )
    try:
        evaluate(env, algo, eval_dataloader)
    finally:
        algo.stop()

    assert env.unwrapped.dataloader is training_dataloader


# --------------------------------------------------------------------------- the trainable


def test_checkpoint_does_not_carry_rc_params(get_model_config, env_config, physical_params, tmp_path, ray_cluster):
    """THE test for PBT correctness.

    Simulates an exploit by hand: trial A trains and checkpoints; trial B has different
    parameters and restores A's checkpoint. Afterwards B must hold A's POLICY WEIGHTS and
    its own PARAMETERS. If the parameters came back from the checkpoint too, every exploit
    would quietly undo the mutation that motivated it and the search would never move.
    """
    low_set, high_set = physical_params
    scaled_low = physical_to_scaled(get_model_config, low_set)
    scaled_high = physical_to_scaled(get_model_config, high_set)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    trial_a = RCPolicyTrainable(config=trial_config(get_model_config, env_config, low_set))
    try:
        trial_a.step()  # move the weights away from their initialisation
        weights_a = copy.deepcopy(trial_a.algo.get_policy().get_weights())
        trial_a.save_checkpoint(str(checkpoint_dir))
    finally:
        trial_a.cleanup()

    trial_b = RCPolicyTrainable(config=trial_config(get_model_config, env_config, high_set))
    try:
        assert_parameters_close(trial_b.rc_scaled, scaled_high)

        trial_b.load_checkpoint(str(checkpoint_dir))

        # ---- the weights came across ----
        weights_b = trial_b.algo.get_policy().get_weights()
        assert set(weights_a) == set(weights_b)
        for key in weights_a:
            np.testing.assert_allclose(weights_a[key], weights_b[key], rtol=1e-6, atol=1e-6)

        # ---- the parameters did NOT ----
        assert_parameters_close(trial_b.rc_scaled, scaled_high)
        for key in RC_PARAM_KEYS:
            assert not np.allclose(np.asarray(scaled_high[key]), np.asarray(scaled_low[key])), (
                f"fixture error: '{key}' is the same in both parameter sets, so this test proves nothing"
            )

        # ---- and the environments the policy actually trains in agree ----
        for params in env_parameters(trial_b.algo):
            assert_parameters_close(params, scaled_high)
    finally:
        trial_b.cleanup()


def test_reset_config_swaps_parameters_in_place(get_model_config, env_config, physical_params, ray_cluster):
    """reset_config is the fast path for an exploit: swap parameters inside the running
    actor instead of tearing it down. The new parameters must reach every environment."""
    low_set, high_set = physical_params
    scaled_high = physical_to_scaled(get_model_config, high_set)

    trial = RCPolicyTrainable(config=trial_config(get_model_config, env_config, low_set))
    try:
        assert trial.reset_config(trial_config(get_model_config, env_config, high_set)) is True

        assert_parameters_close(trial.rc_scaled, scaled_high)
        environments = env_parameters(trial.algo)
        assert environments, "no environment was found to check"
        for params in environments:
            assert_parameters_close(params, scaled_high)
    finally:
        trial.cleanup()


def test_implausible_trial_is_sidelined(get_model_config, env_config, physical_params, ray_cluster):
    """A parameter draw too slow to identify from the data must cost nothing.

    The ranges permit time constants of hundreds of days. Such a trial cannot learn anything
    from a few weeks of data, so rather than spend a population slot on it, it reports the
    penalty without training and PBT exploits it away at the next perturbation.
    """
    _, slow_set = physical_params
    tau_days = slowest_time_constant_days(get_model_config, slow_set)

    trial = RCPolicyTrainable(config=trial_config(get_model_config, env_config, slow_set, max_time_constant_days=tau_days / 2))
    try:
        assert trial.implausible
        assert trial.algo is None, "an implausible trial should not pay to build an algorithm"

        result = trial.step()
        assert result[METRIC] == IMPLAUSIBLE_PENALTY
        assert result["implausible"] is True
        assert result["slowest_tau_days"] == pytest.approx(tau_days)
    finally:
        trial.cleanup()


def test_plausible_trial_trains(get_model_config, env_config, physical_params, ray_cluster):
    """The filter must not sideline a reasonable draw: with a generous threshold the trial
    builds an algorithm and reports a real, finite metric."""
    fast_set, _ = physical_params
    tau_days = slowest_time_constant_days(get_model_config, fast_set)

    trial = RCPolicyTrainable(
        config=trial_config(get_model_config, env_config, fast_set, max_time_constant_days=tau_days * 100)
    )
    try:
        assert not trial.implausible
        result = trial.step()
        assert result["implausible"] is False
        assert np.isfinite(result[METRIC])
        assert result[METRIC] > IMPLAUSIBLE_PENALTY
    finally:
        trial.cleanup()


def test_trainable_records_the_model_it_evaluated(get_model_config, env_config, physical_params, tmp_path, ray_cluster):
    """A record pairs the trial's own RC parameters with the score of that evaluation.

    This pairing is why records are written here at all - a checkpoint carries only the
    policy weights, so it cannot be matched back to its parameters after the run.
    """
    fast_set, _ = physical_params
    trial = RCPolicyTrainable(config=trial_config(get_model_config, env_config, fast_set, model_record_dir=str(tmp_path)))
    try:
        result = trial.step()
    finally:
        trial.cleanup()

    paths = list(tmp_path.glob("*/*.npz"))
    assert len(paths) == 1, "one evaluation should write exactly one record"
    record = load_model_record(paths[0])

    assert record["trial_id"] == trial.trial_id
    assert record["score"] == pytest.approx(result[METRIC])
    assert record["training_iteration"] == 1
    np.testing.assert_allclose(record["param_values"], [fast_set[key] for key in PARAM_KEYS], rtol=1e-5)
    np.testing.assert_allclose(record["cool_w_m2"], fast_set["cool"], rtol=1e-5)
    np.testing.assert_allclose(record["gain_w_m2"], fast_set["gain"], rtol=1e-5)


def test_trainable_rejects_an_out_of_range_record_window(get_model_config, env_config, physical_params, tmp_path):
    fast_set, _ = physical_params
    config = trial_config(get_model_config, env_config, fast_set, model_record_dir=str(tmp_path), model_record_window=99)
    with pytest.raises(ValueError, match="model_record_window"):
        RCPolicyTrainable(config=config)


# --------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_pbt_smoke(get_model_config, env_config, tmp_path, ray_cluster):
    """A real, tiny PBT run: two trials, two iterations, perturbing every iteration.

    Deliberately end-to-end - the pieces above are each correct in isolation, and this is
    what catches them being wired together wrongly (a metric Tune can't find, a config Tune
    can't serialise, a scheduler that never fires).
    """
    tuner = build_tuner(
        model_config=get_model_config,
        env_config=env_config,
        num_samples=2,
        stop={"training_iteration": 2},
        perturbation_interval=1,
        ppo=tiny_ppo_settings(),
        storage_path=str(tmp_path / "ray_results"),
    )
    results = tuner.fit()

    assert len(results) == 2
    assert results.num_errors == 0, "a trial errored - see the Ray logs for the traceback"

    for result in results:
        assert METRIC in result.metrics
        assert np.isfinite(result.metrics[METRIC])

    # The two trials should have drawn different parameters - if they match, the search
    # space isn't being sampled per trial.
    configs = [result.config for result in results]
    assert any(configs[0][key] != configs[1][key] for key in RC_PARAM_KEYS)

    best = results.get_best_result(metric=METRIC, mode="max")
    assert all(key in best.config for key in RC_PARAM_KEYS)


@pytest.mark.slow
def test_initial_population_is_plausible(get_model_config, env_config, tmp_path, ray_cluster):
    """With a threshold set, no trial starts on an implausible draw.

    The threshold is picked so that roughly 3 in 4 prior draws fail it: without screening,
    both trials starting plausible would be a ~6% fluke. With screening, they must - and the
    parameters Tune records must be the screened ones, not a fresh sample.
    """
    threshold = restrictive_threshold(get_model_config)
    tuner = build_tuner(
        model_config=get_model_config,
        env_config=env_config,
        num_samples=2,
        stop={"training_iteration": 1},
        max_time_constant_days=threshold,
        ppo=tiny_ppo_settings(),
        storage_path=str(tmp_path / "ray_results"),
    )
    results = tuner.fit()

    assert results.num_errors == 0, "a trial errored - see the Ray logs for the traceback"
    for result in results:
        assert not result.metrics_dataframe["implausible"].iloc[0], "a trial started on an implausible draw"
        physical = {key: result.config[key] for key in RC_PARAM_KEYS}
        assert slowest_time_constant_days(get_model_config, physical) <= threshold


if __name__ == "__main__":
    pytest.main()
