"""
Model-level plots: what an RC model and its cooling policy did over one evaluation window.

The records drawn here are written during a PBT run by RCPolicyTrainable (set
``model_record_dir``) at evaluation time - the only point where a trial's RC parameters and
the policy weights trained against them are both in hand. The record layout is documented
in rcmodel.optimisation.evaluate().

    from rcmodel.tools.plotting import best_records_over_time, plot_model_record, plot_residual_heatmap

    for record in best_records_over_time("outputs/<run>/model_records", n=4):
        plot_model_record(record)
        plot_residual_heatmap(record)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

MEASURED_COLOR = "#222222"
PREDICTED_COLOR = "#2a78d6"
OUTDOOR_COLOR = "#eb6834"
NODE_COLORS = ("#1baf7a", "#4a3aa7", "#e87ba4", "#eda100")
COOLING_COLOR = "#8fc1f0"


# --------------------------------------------------------------------------- records


def save_model_record(path, record):
    """Write a record (a dict of numpy arrays and scalars) to an .npz file."""
    np.savez(path, **record)


def load_model_record(path):
    """Read a record written by save_model_record(). Scalars come back as Python values."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}


def load_model_records(record_dir):
    """Every record under a run's model_record_dir, as {trial_id: [record, ...]} in time order."""
    records = {}
    for path in sorted(Path(record_dir).glob("*/*.npz")):
        record = load_model_record(path)
        record["path"] = str(path)
        records.setdefault(record["trial_id"], []).append(record)
    if not records:
        raise FileNotFoundError(f"no model records found under {record_dir}")
    for trial_records in records.values():
        trial_records.sort(key=lambda record: record["timestamp"])
    return records


def best_records_over_time(record_dir, n=None):
    """The population's best model as the run went on.

    Walks every evaluation of every trial in wall-clock order, keeping each trial's latest
    record, and notes whichever trial's latest record held the best score at that moment.
    Each record is internally consistent - the parameters, policy and trajectory all come
    from one evaluation - so the sequence shows how the best available fit changed.

    Args:
        record_dir: a run's model_record_dir.
        n: if given, thin the sequence to about n records evenly spaced through the run,
            always keeping the first and last.

    Returns:
        List of records, earliest first, with no immediate repeats.
    """
    every_record = sorted(
        (record for trial_records in load_model_records(record_dir).values() for record in trial_records),
        key=lambda record: record["timestamp"],
    )
    latest = {}
    sequence = []
    for record in every_record:
        latest[record["trial_id"]] = record
        best = max(latest.values(), key=lambda candidate: candidate["score"])
        if not sequence or best is not sequence[-1]:
            sequence.append(best)

    if n is not None and len(sequence) > n:
        keep = np.unique(np.linspace(0, len(sequence) - 1, n).round().astype(int))
        sequence = [sequence[i] for i in keep]
    return sequence


# --------------------------------------------------------------------------- derived values


def _n_latent(record):
    return record["states"].shape[1] - len(record["room_names"])


def _predicted_rooms(record):
    return record["states"][:, _n_latent(record) :]


def _measured_at_prediction_times(record):
    """Measured room temperatures at each simulated time (a subset of the data's times)."""
    return np.column_stack(
        [
            np.interp(record["time"], record["measured_time"], record["measured"][:, i])
            for i in range(len(record["room_names"]))
        ]
    )


def room_rmse(record):
    """Per-room RMSE (degC) between predicted and measured temperature over the window."""
    residual = _predicted_rooms(record) - _measured_at_prediction_times(record)
    return np.sqrt(np.mean(residual**2, axis=0))


def _hours(record, t):
    return (np.asarray(t) - record["time"][0]) / 3600


def _cooling_spans(record):
    """Merged (start, end) spans, in hours, over which the cooling action was on."""
    spans = []
    for action, start, end in zip(record["action"], record["action_start"], record["action_end"], strict=True):
        if not action:
            continue
        start_h, end_h = _hours(record, start), _hours(record, end)
        if spans and np.isclose(spans[-1][1], start_h):
            spans[-1] = (spans[-1][0], end_h)
        else:
            spans.append((start_h, end_h))
    return spans


def _shade_cooling(ax, spans):
    for start, end in spans:
        ax.axvspan(start, end, color=COOLING_COLOR, alpha=0.35, linewidth=0, zorder=0)


def _format_loads(values_w_m2):
    values = np.atleast_1d(values_w_m2)
    if np.allclose(values, values[0]):
        return f"{values[0]:.3g}"
    return f"{values.min():.3g}-{values.max():.3g}"


def _header(record):
    start = np.datetime64(int(record["time"][0]), "s")
    parts = []
    if "trial_id" in record:
        parts.append(f"trial {record['trial_id']}")
    if "training_iteration" in record:
        parts.append(f"iteration {record['training_iteration']}")
    if record.get("score") is not None:
        parts.append(f"score {record['score']:.1f}")
    parts.append(f"eval window {record['window_index']} from {start} (reward {record['window_reward']:.1f})")

    params = "  ".join(
        f"{name}={value:.3g}" for name, value in zip(record["param_names"], record["param_values"], strict=True)
    )
    loads = f"cool {_format_loads(record['cool_w_m2'])} W/m$^2$  gain {_format_loads(record['gain_w_m2'])} W/m$^2$"
    return "   ".join(parts) + "\n" + params + "   " + loads


# --------------------------------------------------------------------------- figures


def plot_model_record(record, ncols=None):
    """Draw one evaluation window: what the model predicted against what was measured.

    Layout, sharing one time axis (hours into the window):

    * top: outdoor temperature and the latent (wall) nodes - walls should lag and damp
      the outdoor swing, so this is the quickest check that the parameters are physical;
    * one panel per room: measured and predicted temperature, with the room's RMSE;
    * bottom: net heat input into the building (gain minus cooling), in W.

    The cooling action is one on/off for the whole building, so cooling-on periods are
    shaded across every panel rather than drawn per room.

    Returns:
        The matplotlib Figure.
    """
    room_names = [str(name) for name in record["room_names"]]
    n_rooms = len(room_names)
    ncols = ncols or min(3, n_rooms)
    nrows = int(np.ceil(n_rooms / ncols))

    # Wide enough for the two-line header even with a single room column.
    fig = plt.figure(figsize=(max(4.8 * ncols + 1.5, 11.0), 2.6 + 2.4 * nrows + 1.8), layout="constrained")
    grid = fig.add_gridspec(nrows + 2, ncols, height_ratios=[1.1] + [1.0] * nrows + [0.7])
    spans = _cooling_spans(record)
    hours = _hours(record, record["time"])

    ax_top = fig.add_subplot(grid[0, :])
    ax_top.plot(hours, record["outdoor"], color=OUTDOOR_COLOR, linewidth=1.4, label="outdoor")
    for k in range(_n_latent(record)):
        ax_top.plot(
            hours, record["states"][:, k], color=NODE_COLORS[k % len(NODE_COLORS)], linewidth=1.2, label=f"wall node {k + 1}"
        )
    _shade_cooling(ax_top, spans)
    ax_top.set_ylabel("°C")
    ax_top.set_title("Outdoor temperature and latent wall nodes", fontsize=10, loc="left")
    ax_top.legend(fontsize=8, loc="upper right", ncol=3)
    ax_top.tick_params(labelbottom=False)

    rmse = room_rmse(record)
    predicted = _predicted_rooms(record)
    measured_hours = _hours(record, record["measured_time"])
    for i, name in enumerate(room_names):
        ax = fig.add_subplot(grid[1 + i // ncols, i % ncols], sharex=ax_top)
        ax.plot(measured_hours, record["measured"][:, i], color=MEASURED_COLOR, linewidth=1.1, label="measured")
        ax.plot(hours, predicted[:, i], color=PREDICTED_COLOR, linewidth=1.4, label="predicted")
        _shade_cooling(ax, spans)
        ax.set_title(f"{name}   RMSE {rmse[i]:.2f} °C", fontsize=9, loc="left")
        if i % ncols == 0:
            ax.set_ylabel("°C")
        if i == 0:
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Patch(color=COOLING_COLOR, alpha=0.35, label="cooling on"))
            ax.legend(handles=handles, fontsize=8, loc="upper right")

    ax_heat = fig.add_subplot(grid[-1, :], sharex=ax_top)
    net = np.sum(record["gain_w"]) - record["action"] * np.sum(record["cool_w"])
    edges = _hours(record, np.concatenate([record["action_start"][:1], record["action_end"]]))
    ax_heat.stairs(net, edges, baseline=None, color=MEASURED_COLOR, linewidth=1.2)
    ax_heat.margins(y=0.2)
    ax_heat.axhline(0, color="0.6", linewidth=0.8)
    _shade_cooling(ax_heat, spans)
    ax_heat.set_ylabel("W")
    ax_heat.set_xlabel("hours into window")
    ax_heat.set_title("Net heat input, all rooms (gain - cooling)", fontsize=10, loc="left")
    ax_heat.set_xlim(hours[0], hours[-1])

    fig.suptitle(_header(record), fontsize=9)
    return fig


def plot_residual_heatmap(record):
    """Predicted minus measured temperature, rooms x time - compact for many rooms.

    Red is the model running warm, blue cold. A thin strip above marks cooling-on periods,
    so errors that line up with the cooling action stand out.

    Returns:
        The matplotlib Figure.
    """
    room_names = [str(name) for name in record["room_names"]]
    residual = _predicted_rooms(record) - _measured_at_prediction_times(record)
    hours = _hours(record, record["time"])
    limit = float(np.max(np.abs(residual))) or 1e-9

    fig = plt.figure(figsize=(12, max(1.6 + 0.35 * len(room_names), 3.2)), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.12, 1.0])
    ax_cool = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=ax_cool)

    _shade_cooling(ax_cool, _cooling_spans(record))
    ax_cool.set_yticks([])
    ax_cool.set_ylabel("cooling", rotation=0, ha="right", va="center", fontsize=8)
    ax_cool.tick_params(labelbottom=False)

    image = ax.imshow(
        residual.T,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extent=(hours[0], hours[-1], len(room_names) - 0.5, -0.5),
        interpolation="nearest",
    )
    ax.set_yticks(range(len(room_names)), room_names, fontsize=8)
    ax.set_xlabel("hours into window")
    ax_cool.set_xlim(hours[0], hours[-1])
    fig.colorbar(image, ax=ax, label="predicted - measured (°C)")
    fig.suptitle(_header(record), fontsize=9)
    return fig
