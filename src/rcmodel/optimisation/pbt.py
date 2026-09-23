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
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.basic_variant import BasicVariantGenerator

import rcmodel.tools
from rcmodel.rc_model import LOAD_KEYS, PARAM_KEYS, RC_PARAM_KEYS

from .evaluation import evaluate, make_update_env_fn

try:  # Ray moved RunConfig/CheckpointConfig around between releases.
    from ray.train import CheckpointConfig, RunConfig
except ImportError:  # pragma: no cover - depends on the installed Ray version
    from ray.tune import CheckpointConfig, RunConfig

logger = logging.getLogger(__name__)

ENV_NAME = "LSIEnv"

#: The metric PBT selects on: mean episode return over the deterministic evaluation split.
METRIC = "eval_return_mean"
MODE = "max"

SECONDS_PER_DAY = 24 * 60**2

#: Score given to a trial whose parameters fail the plausibility filter, so PBT exploits it
#: away at the next perturbation without any training being spent on it. Finite (not -inf)
#: because Tune's quantile bookkeeping has to be able to sort and average it.
IMPLAUSIBLE_PENALTY = -1e9

#: Every PPO training setting a trial's config["ppo"] may hold, with the value used when it is
#: left out. These are RLlib 2.38's own old-API-stack defaults except train_batch_size and
#: minibatch_size, which were set for this problem. Spelled out so a run script can show - and
#: a reader can see - exactly what PPO was trained with. Keys go straight to
#: PPOConfig.training(), except fcnet_hiddens, which goes to its model config.
PPO_DEFAULTS = {
    "lr": 5e-5,  # Adam learning rate. The key PBT can mutate - see build_tuner(ppo_mutations=...).
    "train_batch_size": 256,  # env steps collected per training iteration.
    "minibatch_size": 64,  # SGD minibatch drawn from that batch.
    "num_epochs": 30,  # passes over the batch per iteration - 30 x (256/64) = 120 SGD steps.
    "gamma": 0.99,  # discount; horizon ~ 1/(1-gamma) = 100 steps.
    "lambda_": 1.0,  # GAE lambda. 1.0 = Monte Carlo returns: unbiased, high variance.
    "clip_param": 0.3,  # PPO policy-ratio clip.
    "vf_clip_param": 10.0,  # clamp on the SQUARED value error. The critic gets no gradient
    # wherever |V - return| > sqrt(vf_clip_param), so this must be scaled to the size of the
    # returns (here -MSE summed over an episode: tens to thousands).
    "vf_loss_coeff": 1.0,  # weight of the value loss.
    "entropy_coeff": 0.0,  # entropy bonus. 0 lets the policy collapse to one action early.
    "kl_coeff": 0.2,  # initial KL penalty coefficient (adapted towards kl_target).
    "kl_target": 0.01,
    "grad_clip": None,  # global gradient-norm clip, None for off.
    "fcnet_hiddens": [256, 256],  # policy and value network hidden layers.
}

#: Rollout and resource settings that may also appear in config["ppo"]. These shape how
#: samples are collected, not what is learned from them.
PPO_RUNNER_DEFAULTS = {
    "num_env_runners": 0,  # 0 = sample in the trial's own process; no extra actors.
    "rollout_fragment_length": "auto",
    "num_gpus": 0,
}


def ppo_settings(ppo=None):
    """
    Complete PPO settings: PPO_DEFAULTS and PPO_RUNNER_DEFAULTS overlaid with `ppo`.

    Raises ValueError on a key neither of them knows, so a typo in a run script fails loudly
    instead of silently training with the default.
    """
    ppo = dict(ppo or {})
    unknown = sorted(set(ppo) - set(PPO_DEFAULTS) - set(PPO_RUNNER_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown PPO setting(s) {unknown}. Known: {sorted(PPO_DEFAULTS) + sorted(PPO_RUNNER_DEFAULTS)}.")
    return {**PPO_DEFAULTS, **PPO_RUNNER_DEFAULTS, **ppo}


def ppo_search_space(ppo_mutations):
    """
    Tune Domains for the PPO settings PBT should mutate, from {name: [min, max]}.

    Sampled log-uniformly when min > 0 (learning rates span decades), uniformly otherwise.
    Only numeric PPO_DEFAULTS keys can be mutated.
    """
    space = {}
    for key, (low, high) in (ppo_mutations or {}).items():
        if key not in PPO_DEFAULTS or key == "fcnet_hiddens":
            raise ValueError(f"PBT cannot mutate PPO setting '{key}'. Numeric keys of PPO_DEFAULTS only.")
        space[key] = tune.loguniform(low, high) if low > 0 else tune.uniform(low, high)
    return space


def _register_env_once():
    """Register the environment creator under ENV_NAME.

    Idempotent, and called from Trainable.setup() rather than at import time so that every
    Tune actor process registers it in its own registry when it first builds a trial.
    """
    register_env(ENV_NAME, rcmodel.tools.env_creator)


def physical_to_scaled(model_config, physical):
    """
    Convert physical parameter values to the machine space the model stores: the linear
    map that takes each parameter's configured [min, max] range to [0, 1].

    Values outside the configured range are NOT clipped - they map outside [0, 1] and the
    model runs them as given. PBT's explore step multiplies a value by 0.8 or 1.2 without
    re-clipping, so a trial's config legitimately wanders past the range it was first
    sampled from. Clipping here used to make the model silently run a different value from
    the one the config (and every log and plot) reported. The range only shapes where
    values are sampled from; what keeps a value usable is physical validity (resistances
    and capacitances > 0, loads >= 0), which RCModel.set_parameters() enforces for every
    entry path, not just this one.

    Parameters
    ----------
    model_config : dict
        Supplies a [min, max] range under each key in RC_PARAM_KEYS.
    physical : dict
        {name: value} in physical units. The LOAD_KEYS ("cool", "gain", "solar") may be
        per-room arrays.

    Returns
    -------
    dict
        The same keys in machine space, ready for RCModel.set_parameters().
    """
    scaled = {}
    for key in RC_PARAM_KEYS:
        low, high = model_config[key]
        values = np.asarray(physical[key], dtype=float)
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

    The loads ("cool", "gain", "solar") are sampled uniformly: their ranges start at 0, where
    a log-uniform distribution is undefined, and zero cooling, gain or solar is a physically
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


def sample_plausible_population(model_config, num_samples, max_time_constant_days, log_uniform=True, max_draws=None):
    """
    Draw an initial population whose every member passes the plausibility filter.

    Draws from the same distributions as search_space() and discards any draw whose slowest
    time constant exceeds max_time_constant_days - i.e. it samples the search prior
    truncated to plausible buildings. Left to Tune's own sampling, an implausible draw
    would still get a population slot, but it would report IMPLAUSIBLE_PENALTY and never
    explore its own region before PBT overwrote it with a copy of another trial. Screening
    up front means every slot starts from an independent, plausible draw.

    The draws are passed to Tune as configs (points_to_evaluate), so a trial's parameters
    still come only from its config.

    Parameters
    ----------
    model_config : dict
        Supplies a [min, max] range under each key in RC_PARAM_KEYS.
    num_samples : int
        Population size.
    max_time_constant_days : float
        Plausibility threshold, as for RCPolicyTrainable.
    log_uniform : bool
        As for search_space().
    max_draws : int or None
        Give up after this many draws. Defaults to 100 * num_samples.

    Returns
    -------
    list of dict
        num_samples {name: physical value} dicts covering RC_PARAM_KEYS.
    """
    space = search_space(model_config, log_uniform=log_uniform)
    max_draws = max_draws or 100 * num_samples

    population = []
    draws = 0
    while len(population) < num_samples and draws < max_draws:
        draws += 1
        candidate = {key: float(space[key].sample()) for key in RC_PARAM_KEYS}
        if slowest_time_constant_days(model_config, candidate) <= max_time_constant_days:
            population.append(candidate)

    if len(population) < num_samples:
        raise RuntimeError(
            f"Only {len(population)} of {draws} draws had a slowest time constant within "
            f"max_time_constant_days={max_time_constant_days}; needed {num_samples}. The "
            f"threshold is too tight for these parameter ranges - raise it or narrow the ranges."
        )

    logger.info(
        "Initial population: accepted %d of %d draws (%.0f%%) with slowest time constant <= %s days.",
        num_samples,
        draws,
        100 * num_samples / draws,
        max_time_constant_days,
    )
    return population


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
    ppo                      : dict of PPO settings - see PPO_DEFAULTS and
                               PPO_RUNNER_DEFAULTS for every key and its default. PBT may
                               mutate some of these (build_tuner's ppo_mutations); a trial
                               whose settings change rebuilds its algorithm and keeps its
                               policy weights.
    model_record_dir         : directory or None (default). When set, every evaluation also
                               saves one evaluation window's full trajectory to
                               <model_record_dir>/<trial_id>/<time_ns>.npz - see
                               evaluate() for the contents and rcmodel.tools.plotting to draw
                               it. Records are taken here, where the parameters and the
                               policy weights that produced them are both in hand; a
                               checkpoint holds only the weights, so a model cannot be
                               reliably reassembled from checkpoints after the run.
    model_record_window      : index of the evaluation window to record (default 0). The same
                               window every time, so records compare across trials and time.

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
        self._ppo = ppo_settings(config.get("ppo"))
        self.eval_dataloader = self._make_eval_dataloader(config)
        self._read_record_settings(config)

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

    def _read_record_settings(self, config):
        self.model_record_dir = config.get("model_record_dir")
        self.model_record_window = config.get("model_record_window", 0)
        if self.model_record_dir is None:
            return
        if self.eval_dataloader is None:
            raise ValueError("model_record_dir is set but there is no evaluation split to record from.")
        n_windows = len(self.eval_dataloader)
        if not 0 <= self.model_record_window < n_windows:
            raise ValueError(
                f"model_record_window={self.model_record_window} is out of range: the evaluation "
                f"split has {n_windows} window(s)."
            )

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

        if self.eval_dataloader is not None and self.eval_env is None:
            # Hand the evaluation environment the evaluation loader up front. evaluate()
            # swaps the loader in anyway, but starting on it means that swap is a no-op
            # rather than a second solve of iv_array over a dataset we never use.
            eval_env_config = dict(env_config)
            eval_env_config["dataloader"] = self.eval_dataloader
            self.eval_env = rcmodel.tools.env_creator(eval_env_config)

        if self._pending_weights is not None:
            self._set_weights(self._pending_weights)
            self._pending_weights = None

    def _build_algorithm(self, config, env_config):
        ppo = ppo_settings(config.get("ppo"))
        builder = (
            PPOConfig()
            # The rest of this codebase is written against the old API stack
            # (compute_single_action, get_policy().get_weights(), env_runner_group), so pin
            # it rather than let the default shift underneath us between Ray releases.
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env=ENV_NAME, env_config=env_config, disable_env_checking=True)
            .env_runners(
                num_env_runners=ppo["num_env_runners"],
                rollout_fragment_length=ppo["rollout_fragment_length"],
            )
            .training(
                **{key: ppo[key] for key in PPO_DEFAULTS if key != "fcnet_hiddens"},
                model={"fcnet_hiddens": list(ppo["fcnet_hiddens"])},
            )
            .framework("torch")
            .resources(num_gpus=ppo["num_gpus"])
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
                "ppo_lr": self._ppo["lr"],
            }

        results = self.algo.train()
        train_return = _episode_return_mean(results)

        if self.eval_env is not None and (self.iteration + 1) % self.evaluation_interval == 0:
            record_window = self.model_record_window if self.model_record_dir is not None else None
            reward_list, record = evaluate(self.eval_env, self.algo, self.eval_dataloader, record_window=record_window)
            self._last_eval_return = float(np.mean(reward_list)) if reward_list else None
            if record is not None:
                self._save_record(record)

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
            "ppo_lr": self._ppo["lr"],
        }

    def _save_record(self, record):
        """Write an evaluation record, tagged with what produced it.

        Named by wall-clock time rather than training_iteration, which Ray rewinds when a
        trial is exploited.
        """
        timestamp_ns = time.time_ns()
        record = dict(record)
        record.update(
            {
                "trial_id": self.trial_id,
                "timestamp": timestamp_ns / 1e9,
                "training_iteration": self.iteration + 1,
                "score": self._last_eval_return,  # this evaluation's METRIC
            }
        )
        path = Path(self.model_record_dir) / self.trial_id / f"{timestamp_ns}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        rcmodel.tools.save_model_record(path, record)

    # ---------------------------------------------------------------- checkpointing

    def save_checkpoint(self, checkpoint_dir):
        """Write the policy weights, plus a record of what they were trained against.

        The restorable state is ONLY the policy weights - no RC parameters, see this module's
        docstring. A restore must leave the trial's parameters exactly as its (possibly
        just-mutated) config says.

        TRIAL_STATE_FILE is written alongside for people, not for Tune: it records the RC
        parameters, PPO settings and score that go with these weights, so a checkpoint can be
        picked up after the run (the only other record, result.json, does not say which row a
        checkpoint belongs to). load_checkpoint never reads it.
        """
        weights = self.algo.get_policy().get_weights() if self.algo is not None else self._pending_weights
        with open(Path(checkpoint_dir) / "policy_weights.pkl", "wb") as f:
            pickle.dump({"weights": weights}, f)
        with open(Path(checkpoint_dir) / TRIAL_STATE_FILE, "w") as f:
            json.dump(self._trial_state(), f, indent=2, default=_json_default)
        return None

    def _trial_state(self):
        return {
            "trial_id": self.trial_id,
            "training_iteration": self.iteration,
            "timestamp": time.time(),
            "score": self._last_eval_return,
            "rc_parameters": self.rc_physical,
            "slowest_tau_days": self.tau_days,
            "implausible": self.implausible,
            "ppo": self._ppo,
        }

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

        PPO settings are fixed when an RLlib algorithm is built, so if PBT mutated any (see
        build_tuner's ppo_mutations) the algorithm is rebuilt. Its current weights are carried
        over, so the policy survives even without a restore; after an exploit, the restore
        that follows replaces them with the donor's.
        """
        self.evaluation_interval = new_config.get("evaluation_interval", self.evaluation_interval)
        self._read_record_settings(new_config)

        new_ppo = ppo_settings(new_config.get("ppo"))
        if new_ppo != self._ppo and self.algo is not None:
            self._pending_weights = self.algo.get_policy().get_weights()
            self.algo.stop()
            self.algo = None
        self._ppo = new_ppo

        self._apply_parameters(new_config)
        self._last_eval_return = None  # the old score belonged to the old parameters

        if not self.implausible and self.algo is None:
            # Was sidelined (or its PPO settings changed) and is now viable - build it
            # properly. _apply_parameters has already set the new parameters, and
            # _build_workers applies any weights that arrived while there was no algorithm
            # to put them in.
            self._build_workers(new_config)

        return True

    def cleanup(self):
        if self.algo is not None:
            self.algo.stop()
            self.algo = None


#: Name of the human-readable sidecar save_checkpoint writes next to the policy weights.
TRIAL_STATE_FILE = "trial_state.json"


def _json_default(value):
    """Let json.dump write numpy scalars and arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


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
    perturbation_factors=(1.2, 0.8),
    ppo_mutations=None,
):
    """
    PBT scheduler over the RC parameters, and optionally some PPO settings.

    perturbation_interval is in training iterations. Too small and a trial is judged before
    its policy has adapted to its parameters, which reads a bad policy as bad parameters;
    too large and the search barely moves. It wants to be at least long enough for PPO to
    make visible progress from a fresh set of weights.

    How an exploit explores. At each perturbation the bottom quantile_fraction of trials
    copy a top-quantile trial's config and weights; the top trials themselves are never
    perturbed, so the best set found so far is always kept. Each copied value is then,
    independently:
      * with probability resample_probability, redrawn from its full prior (search_space) -
        a long jump. With k searched values, 1 - (1 - resample_probability)**k of exploits
        resample at least one of them: 94% for the 10 RC values at the default 0.25.
      * otherwise multiplied by one of perturbation_factors - a local step. Repeated steps
        compound (1.2 * 0.8 = 0.96), so the achievable resolution is finer than one factor.
    Lower resample_probability and factors closer to 1 favour fine-tuning around the current
    best; higher favour exploring.

    ppo_mutations: {name: [min, max]} of PPO_DEFAULTS keys to mutate as well, e.g.
    {"lr": [1e-5, 1e-3]}. See ppo_search_space().
    """
    hyperparam_mutations = search_space(model_config, log_uniform=log_uniform)
    if ppo_mutations:
        hyperparam_mutations["ppo"] = ppo_search_space(ppo_mutations)
    return PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        quantile_fraction=quantile_fraction,
        resample_probability=resample_probability,
        perturbation_factors=tuple(perturbation_factors),
        hyperparam_mutations=hyperparam_mutations,
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
    model_record_dir=None,
    model_record_window=0,
    ppo_mutations=None,
    quantile_fraction=0.25,
    resample_probability=0.25,
    perturbation_factors=(1.2, 0.8),
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

    With max_time_constant_days set, the initial population is drawn by
    sample_plausible_population() and handed to Tune as points_to_evaluate, so no slot
    starts on an implausible draw. The threshold still applies to the trials themselves:
    a PBT perturbation that lands on an implausible set is sidelined with
    IMPLAUSIBLE_PENALTY as before.

    model_record_dir / model_record_window: see RCPolicyTrainable. Off by default.

    ppo: PPO settings, see PPO_DEFAULTS. Validated here so a typo fails before any trial starts.

    ppo_mutations, quantile_fraction, resample_probability, perturbation_factors: how PBT
    explores - see build_pbt_scheduler(). A mutated PPO setting's initial value is drawn from
    its range, overriding anything given for it in `ppo`.

    Checkpoints and paused trials: an exploited trial restores from a copy of the donor's
    checkpoint. If there are more trials than concurrent slots, trials get paused and queued,
    and by the time one resumes the donor may have saved newer checkpoints and pruned the
    one it needs - it then errors out. Run with num_samples no larger than the number of
    trials that fit at once, and keep enough checkpoints (num_to_keep=None keeps them all).
    """
    ppo = ppo_settings(ppo)
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
            "ppo": {**ppo, **ppo_search_space(ppo_mutations)},
            "model_record_dir": None if model_record_dir is None else str(model_record_dir),
            "model_record_window": model_record_window,
        }
    )

    search_alg = None
    if max_time_constant_days is not None:
        # Each point replaces one of num_samples, and the RC keys it sets override the
        # search_space Domains - everything else still comes from param_space.
        search_alg = BasicVariantGenerator(
            points_to_evaluate=sample_plausible_population(
                model_config, num_samples, max_time_constant_days, log_uniform=log_uniform
            )
        )

    return tune.Tuner(
        RCPolicyTrainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=METRIC,
            mode=MODE,
            scheduler=build_pbt_scheduler(
                model_config,
                perturbation_interval=perturbation_interval,
                quantile_fraction=quantile_fraction,
                resample_probability=resample_probability,
                log_uniform=log_uniform,
                perturbation_factors=perturbation_factors,
                ppo_mutations=ppo_mutations,
            ),
            search_alg=search_alg,
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

    Returns a (physical, scaled) pair: the physical values for reporting, and the
    machine-space dict ready to hand to RCModel.set_parameters() or
    model_config["parameters"].
    """
    best = results.get_best_result(metric=METRIC, mode=MODE)
    physical = {key: best.config[key] for key in RC_PARAM_KEYS}
    return physical, physical_to_scaled(best.config["model_config"], physical)


__all__ = [
    "ENV_NAME",
    "IMPLAUSIBLE_PENALTY",
    "METRIC",
    "MODE",
    "PPO_DEFAULTS",
    "PPO_RUNNER_DEFAULTS",
    "TRIAL_STATE_FILE",
    "RCPolicyTrainable",
    "best_parameters",
    "build_pbt_scheduler",
    "build_tuner",
    "physical_to_scaled",
    "ppo_search_space",
    "ppo_settings",
    "sample_plausible_population",
    "search_space",
    "slowest_time_constant_days",
]
