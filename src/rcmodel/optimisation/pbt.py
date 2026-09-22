"""
Population Based Training of a cooling policy over randomly drawn RC parameters.

The idea
--------
The RC parameters used to be fitted by gradient descent while an RL policy was trained
against them in alternation. Every time the parameters moved, the policy was training
against a different building, so it was continually invalidated and effectively thrown away.

Here the parameters are drawn at random instead and held FIXED for the life of a trial, so
each policy trains against a stationary environment. PBT supplies the search: trials that
explain the measured data badly copy the parameters *and the policy weights* of trials that
explain it well, with the parameters perturbed. Nothing is discarded - a good policy travels
with the parameters it was trained for.

The invariant everything rests on
---------------------------------
    RC parameters come from the trial's config, and ONLY from the config.
    The checkpoint carries policy weights and nothing else.

A PBT exploit copies a source trial's checkpoint into a target trial and mutates the
target's config. If RC parameters were also inside the checkpoint, restoring it would
overwrite the freshly mutated parameters with the source trial's old ones and the search
would silently do nothing - every trial would drift toward whichever parameters happened to
checkpoint last. That is why this module defines its own Trainable rather than handing
RLlib's PPO straight to Tune: PPO's own checkpoint restores its saved config, env_config
included. test_checkpoint_does_not_carry_rc_params pins the invariant down.
"""

import copy
import pickle
from pathlib import Path

import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

import rcmodel.tools
from rcmodel.rc_model import LOAD_KEYS, PARAM_KEYS, RC_PARAM_KEYS

from .evaluation import evaluate, make_update_env_fn

try:  # Ray moved RunConfig/CheckpointConfig around between releases.
    from ray.train import CheckpointConfig, RunConfig
except ImportError:  # pragma: no cover - depends on the installed Ray version
    from ray.tune import CheckpointConfig, RunConfig

ENV_NAME = "LSIEnv"

#: The metric PBT selects on: mean episode return over the deterministic evaluation split.
METRIC = "eval_return_mean"
MODE = "max"

SECONDS_PER_DAY = 24 * 60**2

#: Score given to a trial whose parameters fail the plausibility filter, so PBT exploits it
#: away at the next perturbation without any training being spent on it. Finite (not -inf)
#: because Tune's quantile bookkeeping has to be able to sort and average it.
IMPLAUSIBLE_PENALTY = -1e9


def _register_env_once():
    """Register the environment creator under ENV_NAME.

    Idempotent, and called from Trainable.setup() rather than at import time so that every
    Tune actor process registers it in its own registry when it first builds a trial.
    """
    register_env(ENV_NAME, rcmodel.tools.env_creator)


def physical_to_scaled(model_config, physical):
    """
    Convert physical parameter values to the 0-1 machine space the model stores, clipping
    each one to its configured range.

    The clipping is not cosmetic. PBT perturbs a value by multiplying it (0.8x / 1.2x by
    default), which walks it straight out of its range sooner or later, and
    InputScaling.minmaxscale ASSERTS that its input is in range - so without clipping an
    exploit would eventually kill a trial with an AssertionError rather than exploring its
    boundary.

    Parameters
    ----------
    model_config : dict
        Supplies a [min, max] range under each key in RC_PARAM_KEYS.
    physical : dict
        {name: value} in physical units. "cool" and "gain" may be per-room arrays.

    Returns
    -------
    dict
        The same keys, scaled to 0-1, ready for RCModel.set_parameters().
    """
    scaled = {}
    for key in RC_PARAM_KEYS:
        low, high = model_config[key]
        values = np.clip(np.asarray(physical[key], dtype=float), low, high)
        # A degenerate range (min == max) means the parameter is pinned, not searched.
        span = high - low
        normalised = (values - low) / span if span > 0 else np.zeros_like(values)
        scaled[key] = normalised.item() if normalised.ndim == 0 or normalised.size == 1 else normalised
    return scaled


def search_space(model_config, log_uniform=True):
    """
    Tune search space over the RC parameters, in physical units.

    Capacitances and resistances are sampled LOG-uniformly by default. Their ranges span
    orders of magnitude (C1 covers 1e5 to 1e8), so uniform sampling would put roughly 99% of
    draws in the top decade and the search would never see a small-capacitance building at
    all.

    The loads ("cool", "gain") are sampled uniformly: their ranges start at 0, where a
    log-uniform distribution is undefined, and zero cooling or zero gain is a physically
    meaningful draw that the search should be able to make.

    Parameters
    ----------
    model_config : dict
        Supplies a [min, max] range under each key in RC_PARAM_KEYS.
    log_uniform : bool
        Set False to sample every parameter uniformly instead.

    Returns
    -------
    dict
        {name: tune Domain}, usable both as param_space entries and as PBT
        hyperparam_mutations.
    """
    space = {}
    for key in PARAM_KEYS:
        low, high = model_config[key]
        if log_uniform and low > 0:
            space[key] = tune.loguniform(low, high)
        else:
            space[key] = tune.uniform(low, high)
    for key in LOAD_KEYS:
        low, high = model_config[key]
        space[key] = tune.uniform(low, high)
    return space


def slowest_time_constant_days(model_config, physical_params):
    """
    Slowest time constant, in days, of a parameter set - without building an environment.

    Used by the plausibility filter. Builds a bare model (no dataset, so no iv_array is
    solved) purely to assemble A and read its eigenvalues.
    """
    probe_config = copy.deepcopy(model_config)
    probe_config["parameters"] = physical_to_scaled(model_config, physical_params)
    model = rcmodel.tools.model_creator(probe_config)
    model._build_matrices()
    return model.slowest_time_constant() / SECONDS_PER_DAY


class RCPolicyTrainable(tune.Trainable):
    """
    One PBT trial: a fixed set of RC parameters, and a PPO policy learning against them.

    Config keys
    -----------
    <RC_PARAM_KEYS>          : the searched parameters, in physical units. PBT mutates these.
    model_config             : dict as model_creator() expects. Its "parameters" entry is
                               overwritten from the searched values, so whatever it holds is
                               ignored.
    env_config               : dict as env_creator() expects, minus the model - see
                               _env_config().
    eval_dataloader          : deterministic loader used for the selection metric.
    max_time_constant_days   : float or None. Trials whose slowest mode is slower than this
                               are sidelined - see the plausibility filter below. None
                               disables the filter.
    evaluation_interval      : run the evaluation every N iterations (default 1).
    ppo                      : dict of PPO settings, see _build_algorithm().

    The plausibility filter
    -----------------------
    The parameter ranges legally permit R*C products with time constants of hundreds of
    days, and a building that slow cannot be identified from a few weeks of data - a trial
    holding such a draw burns compute to learn nothing. Rather than crash or silently waste
    the slot, such a trial reports IMPLAUSIBLE_PENALTY without training, so PBT exploits it
    away at the next perturbation and the slot goes back into the search. The threshold is a
    config value because what counts as "too slow" depends on how much data you have.
    """

    def setup(self, config):
        _register_env_once()

        self.algo = None
        self.eval_env = None
        self._pending_weights = None
        self._last_eval_return = None

        self.evaluation_interval = config.get("evaluation_interval", 1)
        self.eval_dataloader = self._make_eval_dataloader(config)

        self._apply_parameters(config)

        if not self.implausible:
            self._build_workers(config)

    @staticmethod
    def _make_eval_dataloader(config):
        """The deterministic loader the selection metric is computed over.

        Built here from plain config values rather than passed in as an object, so a trial's
        config stays serialisable - Tune deep-copies it between trials and logs it.
        """
        if config.get("eval_dataloader") is not None:
            return config["eval_dataloader"]
        data_config = config.get("env_config", {}).get("data_config") or config.get("data_config")
        if data_config is None:
            return None
        _, eval_dataloader = rcmodel.tools.make_dataloaders(data_config)
        return eval_dataloader

    # ---------------------------------------------------------------- parameters

    def _apply_parameters(self, config):
        """Read the searched parameters out of the config and push them everywhere they live.

        This is the only place RC parameters enter the trial. Called from setup() and again
        from reset_config() after a PBT exploit.
        """
        self.model_config = copy.deepcopy(config["model_config"])
        self.rc_physical = {key: config[key] for key in RC_PARAM_KEYS}
        self.rc_scaled = physical_to_scaled(self.model_config, self.rc_physical)
        self.model_config["parameters"] = self.rc_scaled

        self.tau_days = slowest_time_constant_days(config["model_config"], self.rc_physical)
        max_tau = config.get("max_time_constant_days")
        self.implausible = max_tau is not None and self.tau_days > max_tau

        # Push the new parameters to anything already built. Each of these rebuilds A/B and
        # re-solves iv_array for the new parameters (see LSIEnv.update_from_config).
        if self.algo is not None:
            self.algo.env_runner_group.foreach_worker(make_update_env_fn({"rc_parameters": self.rc_scaled}))
        if self.eval_env is not None:
            self.eval_env.unwrapped.update_from_config({"rc_parameters": self.rc_scaled})

    def _env_config(self, config):
        env_config = copy.deepcopy(config.get("env_config", {}))
        env_config["model_config"] = self.model_config
        # Belt and braces: the model is built from model_config["parameters"] above, and
        # this makes the environment apply the same values again on its first update. A
        # live model object must never appear here - Tune deep-copies configs between
        # trials, so parameters would stop tracking the config.
        env_config["rc_parameters"] = self.rc_scaled
        env_config.pop("RC_model", None)
        return env_config

    def _build_workers(self, config):
        env_config = self._env_config(config)
        self.algo = self._build_algorithm(config, env_config)

        if self.eval_dataloader is not None:
            # Hand the evaluation environment the evaluation loader up front. evaluate()
            # swaps the loader in anyway, but starting on it means that swap is a no-op
            # rather than a second solve of iv_array over a dataset we never use.
            eval_env_config = dict(env_config)
            eval_env_config["render_mode"] = None
            eval_env_config["dataloader"] = self.eval_dataloader
            self.eval_env = rcmodel.tools.env_creator(eval_env_config)

        if self._pending_weights is not None:
            self._set_weights(self._pending_weights)
            self._pending_weights = None

    def _build_algorithm(self, config, env_config):
        ppo = config.get("ppo", {})
        builder = (
            PPOConfig()
            # The rest of this codebase is written against the old API stack
            # (compute_single_action, get_policy().get_weights(), env_runner_group), so pin
            # it rather than let the default shift underneath us between Ray releases.
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env=ENV_NAME, env_config=env_config, disable_env_checking=True)
            .env_runners(
                num_env_runners=ppo.get("num_env_runners", 0),
                rollout_fragment_length=ppo.get("rollout_fragment_length", "auto"),
            )
            .training(
                train_batch_size=ppo.get("train_batch_size", 256),
                minibatch_size=ppo.get("minibatch_size", 64),
                lr=ppo.get("lr", 5e-5),
                gamma=ppo.get("gamma", 0.99),
            )
            .framework("torch")
            .resources(num_gpus=ppo.get("num_gpus", 0))
        )
        return builder.build()

    # ---------------------------------------------------------------- training

    def step(self):
        if self.implausible:
            # No training at all - the point of the filter is not to spend the slot.
            return {
                METRIC: IMPLAUSIBLE_PENALTY,
                "train_return_mean": IMPLAUSIBLE_PENALTY,
                "slowest_tau_days": self.tau_days,
                "implausible": True,
            }

        results = self.algo.train()
        train_return = _episode_return_mean(results)

        if self.eval_env is not None and (self.iteration + 1) % self.evaluation_interval == 0:
            reward_list, _ = evaluate(self.eval_env, self.algo, self.eval_dataloader)
            self._last_eval_return = float(np.mean(reward_list)) if reward_list else None

        # PBT needs METRIC present on EVERY result, so carry the last evaluation forward on
        # iterations that didn't run one. Falling back to the train return keeps the very
        # first iterations comparable when evaluation_interval > 1.
        eval_return = self._last_eval_return
        if eval_return is None:
            eval_return = train_return

        return {
            METRIC: eval_return,
            "train_return_mean": train_return,
            "slowest_tau_days": self.tau_days,
            "implausible": False,
        }

    # ---------------------------------------------------------------- checkpointing

    def save_checkpoint(self, checkpoint_dir):
        """Write ONLY the policy weights.

        Deliberately no RC parameters: see this module's docstring. A restore must leave the
        trial's parameters exactly as its (possibly just-mutated) config says.
        """
        weights = self.algo.get_policy().get_weights() if self.algo is not None else self._pending_weights
        with open(Path(checkpoint_dir) / "policy_weights.pkl", "wb") as f:
            pickle.dump({"weights": weights}, f)
        return None

    def load_checkpoint(self, checkpoint):
        payload = checkpoint
        if not isinstance(payload, dict) or "weights" not in payload:
            # Ray hands back the checkpoint directory when save_checkpoint returned None.
            with open(Path(checkpoint) / "policy_weights.pkl", "rb") as f:
                payload = pickle.load(f)
        self._set_weights(payload["weights"])

    def _set_weights(self, weights):
        if weights is None:
            return
        if self.algo is None:
            # An implausible trial has no algorithm yet. Hold the weights so they are applied
            # if a later exploit gives this trial a plausible parameter set.
            self._pending_weights = weights
            return
        self.algo.set_weights({"default_policy": weights})
        self.algo.env_runner_group.sync_weights()

    def reset_config(self, new_config):
        """Adopt a mutated config in place, so a PBT exploit doesn't rebuild the actor.

        Tune calls this before restoring the checkpoint, which is the order this needs:
        parameters first, then the weights that come with them.
        """
        self.evaluation_interval = new_config.get("evaluation_interval", self.evaluation_interval)
        self._apply_parameters(new_config)
        self._last_eval_return = None  # the old score belonged to the old parameters

        if not self.implausible and self.algo is None:
            # Was sidelined, now viable - build it properly. _apply_parameters has already
            # set the new parameters, and _build_workers applies any weights that arrived
            # while there was no algorithm to put them in.
            self._build_workers(new_config)

        return True

    def cleanup(self):
        if self.algo is not None:
            self.algo.stop()
            self.algo = None


def _episode_return_mean(results):
    """Pull the mean episode return out of an RLlib result dict, tolerating key moves."""
    for path in (("env_runners", "episode_return_mean"), ("sampler_results", "episode_reward_mean")):
        node = results
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None and not (isinstance(node, float) and np.isnan(node)):
            return float(node)
    return float(results.get("episode_reward_mean", float("nan")))


def build_pbt_scheduler(
    model_config,
    perturbation_interval=4,
    quantile_fraction=0.25,
    resample_probability=0.25,
    log_uniform=True,
):
    """
    PBT scheduler over the RC parameters.

    perturbation_interval is in training iterations. Too small and a trial is judged before
    its policy has adapted to its parameters, which reads a bad policy as bad parameters;
    too large and the search barely moves. It wants to be at least long enough for PPO to
    make visible progress from a fresh set of weights.
    """
    return PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        quantile_fraction=quantile_fraction,
        resample_probability=resample_probability,
        hyperparam_mutations=search_space(model_config, log_uniform=log_uniform),
    )


def build_tuner(
    model_config,
    env_config,
    eval_dataloader=None,
    num_samples=8,
    stop=None,
    max_time_constant_days=None,
    evaluation_interval=1,
    ppo=None,
    perturbation_interval=4,
    log_uniform=True,
    storage_path=None,
    checkpoint_config=None,
):
    """
    Assemble a Tuner running RCPolicyTrainable under PBT.

    num_samples is the population size. PBT needs enough trials for its quantiles to mean
    something - below about 4 the top and bottom quartiles are the same one or two trials -
    and this is a 9-dimensional search, so more is better where the hardware allows.

    eval_dataloader is normally left as None: each trial builds its own from
    env_config["data_config"], which keeps the config plain serialisable data. Pass one
    explicitly only to score against a split make_dataloaders doesn't produce.

    reuse_actors=True is what makes reset_config() worth having: an exploit swaps parameters
    inside the running actor instead of tearing it down and rebuilding the environment.
    """
    if eval_dataloader is None and not env_config.get("data_config"):
        raise ValueError(
            "Trials need a deterministic evaluation split to be scored on. Either put a "
            "'data_config' in env_config (see make_dataloaders) or pass eval_dataloader "
            "explicitly - otherwise the metric silently falls back to the noisy training "
            "return and PBT selects on sampling luck."
        )

    param_space = dict(search_space(model_config, log_uniform=log_uniform))
    param_space.update(
        {
            "model_config": model_config,
            "env_config": env_config,
            "eval_dataloader": eval_dataloader,
            "max_time_constant_days": max_time_constant_days,
            "evaluation_interval": evaluation_interval,
            "ppo": ppo or {},
        }
    )

    return tune.Tuner(
        RCPolicyTrainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=METRIC,
            mode=MODE,
            scheduler=build_pbt_scheduler(model_config, perturbation_interval=perturbation_interval, log_uniform=log_uniform),
            num_samples=num_samples,
            reuse_actors=True,
        ),
        run_config=RunConfig(
            stop=stop or {"training_iteration": 20},
            storage_path=storage_path,
            checkpoint_config=checkpoint_config
            or CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute=METRIC,
                checkpoint_score_order=MODE,
            ),
        ),
    )


def best_parameters(results):
    """
    Physical RC parameters of the best trial in a finished Tuner run.

    Returns a (physical, scaled) pair: the physical values for reporting, and the 0-1 scaled
    dict ready to hand to RCModel.set_parameters() or model_config["parameters"].
    """
    best = results.get_best_result(metric=METRIC, mode=MODE)
    physical = {key: best.config[key] for key in RC_PARAM_KEYS}
    return physical, physical_to_scaled(best.config["model_config"], physical)


__all__ = [
    "ENV_NAME",
    "IMPLAUSIBLE_PENALTY",
    "METRIC",
    "MODE",
    "RCPolicyTrainable",
    "best_parameters",
    "build_pbt_scheduler",
    "build_tuner",
    "physical_to_scaled",
    "search_space",
    "slowest_time_constant_days",
]
