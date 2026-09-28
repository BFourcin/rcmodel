"""
Stage A of a two-stage fit: identify the envelope from the periods the building is free-floating.

A PBT search fits the RC parameters and the cooling policy together, and the pair can trade against
each other - cooling too hard, and making up the heat elsewhere. When the building's HVAC schedule is
known, the stretches outside it (nights, weekends) are driven by the weather alone: no cooling, no
policy. Fitting the envelope there first, then letting PBT search the cooling (and only a narrow band
around the envelope) removes that trade.

    fitted = fit_free_float(model_config, data_config)          # stage A
    stage_b = narrow_ranges(model_config, fitted["physical"])    # stage B: build_tuner(stage_b, ...)

The fit is a direct least-squares search (Nelder-Mead in log space, bounded to the configured ranges),
like a hand-tuned refit: each free-float stretch is an episode restarted from the measured room
temperatures, with the envelope (and any room mass) nodes warmed up over the history exactly as in
training, and the model run with the cooling action off.

What free-float data does and doesn't identify - checked against EnergyPlus TrialA_1Zone
(using_rcmodel/energyplus_1zone_rc_check.ipynb, §10e-f):
- With room_mass on, PIN the mass node (C_rm_mass, R_rm_mass) at design values for this stage. Its state
  at each stretch start is estimated by filtering the measured air, which assumes the mass exchanges heat
  only with the air; a real floor is also warmed radiatively by the walls, so the estimate starts too
  cold, and a free mass lets the fit trade it against the envelope - it halved UA there. Pinned, UA came
  out within ~25 % of the building's.
- The resistances and k_sa (UA and the solar gain) are identified; the envelope capacitances are not (C1
  ran to its bound whatever its start). Narrow only the former for stage B.
"""

import numpy as np
import torch
from scipy.optimize import minimize

import rcmodel.tools
from rcmodel.rc_model import RC_PARAM_KEYS, param_range

from .pbt import physical_to_scaled, pinned_parameters

# What stage A searches by default: the envelope and the room's thermal mass - the parameters the
# weather alone drives. Cooling, setpoint and gains belong to stage B.
ENVELOPE_KEYS = ("R1", "R2", "R3", "C1", "C2", "k_sa", "C_rm", "C_rm_mass", "R_rm_mass", "Rin")


def free_float_segments(time, free, min_rows, max_rows=None, exclude=None):
    """Index arrays of the contiguous stretches where `free` is True (and `exclude` is not), at least
    min_rows long; stretches longer than max_rows are cut into max_rows-long pieces (the last may be
    shorter, but not below min_rows)."""
    free = np.asarray(free, dtype=bool).copy()
    if exclude is not None:
        free &= ~np.asarray(exclude, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], free.astype(int), [0]])))
    segments = []
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        step = max_rows or (stop - start)
        for a in range(start, stop, step):
            b = min(a + step, stop)
            if b - a >= min_rows:
                segments.append(np.arange(a, b))
    return segments


def _evaluation_rows(data_config, n_rows):
    """Rows the evaluation split uses, so stage A never fits on data the search is scored on."""
    split = data_config.get("eval_split", "tail")
    held_out = np.zeros(n_rows, dtype=bool)
    if split == "tail":
        held_out[int(0.8 * n_rows) :] = True
    else:
        size = int(data_config["sample_size"])
        for block in rcmodel.tools.interleaved_blocks(n_rows, size, int(split["every"])):
            held_out[block * size : (block + 1) * size] = True
    return held_out


def _start_values(model_config, keys, start):
    values = {}
    for key in keys:
        if start and key in start:
            values[key] = float(start[key])
            continue
        low, high = param_range(model_config, key)
        values[key] = float(np.sqrt(low * high)) if low > 0 else (low + high) / 2
    return values


def fit_free_float(
    model_config,
    data_config,
    keys=None,
    start=None,
    min_hours=12,
    max_hours=72,
    warmup_hours=72,
    score_after_hours=0.0,
    maxfev=800,
):
    """
    Fit the envelope parameters on the free-floating stretches of the training data.

    Parameters
    ----------
    model_config : dict        as model_creator() takes; must include "hvac_schedule" (see
                               rcmodel.tools.free_float_mask).
    data_config : dict         as make_dataloaders() takes; its evaluation split is left out.
    keys : list or None        parameters to fit. Default: ENVELOPE_KEYS that are not pinned.
    start : dict or None       physical starting values (e.g. hand-derived); every other parameter,
                               fitted or not, starts at its range's (geometric) middle.
    min_hours, max_hours : float  shortest stretch used, and the longest episode (longer stretches
                               are cut up, so a long weekend doesn't dominate).
    score_after_hours : float  each stretch is simulated from its start but only scored after this long,
                               so the error in its estimated starting state has time to decay first.
    warmup_hours : float       nothing in the record's first warmup_hours is fitted on: the envelope
                               nodes start from a steady-state guess at the first row (see
                               get_iv_array) and need a few of their time constants to settle.
    maxfev : int               Nelder-Mead evaluation budget. The search stays inside each parameter's
                               configured range.

    Returns
    -------
    dict with "physical" (every RC parameter, physical units: the fitted ones and the fixed ones),
    "fitted_keys", "rmse" (degC over the stretches), "n_segments", "n_rows", and "result" (scipy's).
    """
    schedule = model_config.get("hvac_schedule")
    if schedule is None:
        raise ValueError("fit_free_float needs model_config['hvac_schedule'] to know when the building free-floats.")
    pinned = pinned_parameters(model_config)
    keys = [k for k in (keys or ENVELOPE_KEYS) if k not in pinned]
    if not keys:
        raise ValueError("Nothing to fit: every requested parameter is pinned.")

    path = rcmodel.tools.sort_data(str(data_config["csv_path"]), data_config.get("dt", 30))
    dataset = rcmodel.tools.BuildingTemperatureDataset(path, sample_size=1, all=True)
    t, temps = dataset.get_all_data()
    t_np = t.numpy()
    n_rooms = len(model_config["room_names"])
    measured = temps[:, :n_rooms].numpy().astype(np.float64)

    dt = float(np.median(np.diff(t_np)))
    free = rcmodel.tools.free_float_mask(t_np, schedule)
    excluded = _evaluation_rows(data_config, len(t_np))
    excluded[: int(warmup_hours * 3600 / dt)] = True
    segments = free_float_segments(t_np, free, int(min_hours * 3600 / dt), int(max_hours * 3600 / dt), exclude=excluded)
    if not segments:
        raise ValueError(f"No free-float stretch of at least {min_hours} h outside the warm-up and the evaluation split.")
    whole = rcmodel.tools.BuildingTemperatureDataset(path, sample_size=len(t_np), all=True)
    skip = int(score_after_hours * 3600 / dt)
    if skip >= int(min_hours * 3600 / dt):
        raise ValueError("score_after_hours must be shorter than min_hours, or nothing would be scored.")

    fixed = _start_values(model_config, [k for k in RC_PARAM_KEYS if k not in keys], start)
    fixed.update(pinned)
    x0 = np.log(np.maximum([_start_values(model_config, [k], start)[k] for k in keys], 1e-12))

    def physical(x):
        return {**fixed, **dict(zip(keys, np.exp(x), strict=True))}

    def rmse(x):
        values = physical(x)
        try:
            model = rcmodel.tools.model_creator({**model_config, "parameters": physical_to_scaled(model_config, values)})
            model.setup(whole)
        except (ValueError, AssertionError, np.linalg.LinAlgError):
            return 1e3
        squared, count = 0.0, 0
        with torch.no_grad():
            for seg in segments:
                model.iv = model.iv_array(t[seg[0]])
                pred = model(t[seg], action=0).squeeze(-1)[:, 2 : 2 + n_rooms].numpy()[skip:]
                squared += float(np.sum((pred - measured[seg][skip:]) ** 2))
                count += pred.size
        value = np.sqrt(squared / count)
        return value if np.isfinite(value) else 1e3

    # Bounded to the configured ranges: free-float data alone identifies the envelope only weakly, and
    # unbounded, the fit happily finds degenerate buildings (e.g. a brick node so heavy it becomes a
    # constant-temperature reservoir) that score well on the stretches and badly as a building.
    bounds = [tuple(np.log(np.maximum(param_range(model_config, k), 1e-12))) for k in keys]
    x0 = np.clip(x0, [lo for lo, _ in bounds], [hi for _, hi in bounds])
    result = minimize(rmse, x0, method="Nelder-Mead", bounds=bounds, options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-4})
    return {
        "physical": physical(result.x),
        "fitted_keys": keys,
        "rmse": float(result.fun),
        "start_rmse": float(rmse(x0)),
        "n_segments": len(segments),
        "n_rows": int(sum(len(s) for s in segments)),
        "result": result,
    }


def narrow_ranges(model_config, fitted, keys=None, factor=1.5):
    """A copy of model_config with each of `keys` (default: every fitted parameter that has a range)
    searched only within [value / factor, value * factor] - stage B's starting point after stage A.

    The narrowed range may extend past the original one: the fitted value is what the data said.
    """
    narrowed = dict(model_config)
    for key in keys or fitted:
        if key not in RC_PARAM_KEYS or key in pinned_parameters(model_config):
            continue
        value = float(fitted[key])
        if value > 0:
            narrowed[key] = [value / factor, value * factor]
    return narrowed


__all__ = ["ENVELOPE_KEYS", "fit_free_float", "free_float_segments", "narrow_ranges"]
