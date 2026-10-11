"""
A parametric cooling controller: cooling is available inside weekly time windows, optionally only while the
room is above a setpoint.

The alternative to a PPO policy when the building's controller is (or is assumed to be) a schedule. It has a
handful of interpretable parameters - the windows' hours, and a setpoint in switch mode - which can be searched
jointly with the RC parameters by any deterministic optimiser, instead of a policy network that has to be
retrained for every parameter set and can compensate for wrong parameters.

It is duck-typed like an RLlib algorithm for evaluate(): compute_single_action(obs, explore) -> 0/1. It reads
the time (and room temperature) from the unwrapped environment, which evaluate() leaves at the start of the step
it is about to take, so it works whatever the observation wrapper feeds a policy.

With cooling_mode="thermostat" the model's T_set does the thermostat's job and the policy is only the
availability gate - leave setpoint=None. With cooling_mode="switch" the action IS the cooling, so pass a
setpoint to get a bang-bang thermostat inside the windows.

    policy = SchedulePolicy(env, [{"weekdays": [0, 1, 2, 3, 4], "start": 7, "end": 19}])
    returns, _ = evaluate(env, policy, eval_dataloader)
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

SECONDS_PER_DAY = 86400
_OFFSETS = {}  # (time zone, UTC hour since 1970) -> that zone's UTC offset in seconds


def utc_offset(value):
    """A clock for schedules: a fixed offset from UTC in hours (a number), or an IANA time zone name such as
    "Europe/London", whose offset follows daylight saving. Numbers come back as floats, names as they are (checked)."""
    if isinstance(value, str):
        ZoneInfo(value)  # raises for an unknown zone
        return value
    return float(value)


def local_seconds(time, utc_offset_hours=0.0):
    """Unix times `time` (s, UTC) moved onto the local wall clock: seconds since 1970-01-01 00:00 LOCAL, so that
    floor(local / 86400) is the local day and local % 86400 the local time of day. utc_offset_hours is a number of
    hours or a time zone name (see utc_offset); a zone's offset is looked up once per UTC hour and cached, which is
    exact for zones that change clocks on the hour in UTC (all of Europe)."""
    t = np.asarray(time, dtype=np.float64)
    if not isinstance(utc_offset_hours, str):
        return t + 3600 * float(utc_offset_hours)
    zone = ZoneInfo(utc_offset_hours)
    hours = np.floor(t / 3600).astype(np.int64)
    unique, inverse = np.unique(hours.ravel(), return_inverse=True)
    offsets = np.empty(len(unique))
    for i, hour in enumerate(unique.tolist()):
        key = (utc_offset_hours, hour)
        if key not in _OFFSETS:
            _OFFSETS[key] = datetime.fromtimestamp(hour * 3600, tz=zone).utcoffset().total_seconds()
        offsets[i] = _OFFSETS[key]
    return t + offsets[inverse].reshape(t.shape)


def _hours(value):
    """Hours since local midnight from a number of hours or an "HH:MM" string (as in hvac_schedule)."""
    if isinstance(value, str):
        h, m = (int(x) for x in value.split(":"))
        return h + m / 60
    return float(value)


def normalise_windows(windows):
    """[{weekdays, start, end}] with start/end as float hours; accepts one hvac_schedule-style dict too."""
    if isinstance(windows, dict):
        windows = [windows]
    return [
        {"weekdays": [int(d) for d in w["weekdays"]], "start": _hours(w["start"]), "end": _hours(w["end"])} for w in windows
    ]


def schedule_available(time, windows, utc_offset_hours=0.0):
    """True at unix times `time` (s, UTC) inside any window. Vectorised.

    A window is {"weekdays": [Monday=0 ...], "start": h, "end": h}: available from start (inclusive) to end
    (exclusive), local time = UTC + utc_offset_hours, a number of hours or a time zone name such as
    "Europe/London" (daylight saving followed). start >= end is an empty window, so a search can switch
    a window off by collapsing it. Same convention as free_float_mask (which is its complement for one window).
    """
    local = local_seconds(time, utc_offset_hours)
    weekday = (np.floor(local / SECONDS_PER_DAY).astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday
    hour = (local % SECONDS_PER_DAY) / 3600
    available = np.zeros(np.shape(local), dtype=bool)
    for window in normalise_windows(windows):
        available |= np.isin(weekday, window["weekdays"]) & (hour >= window["start"]) & (hour < window["end"])
    return available


class SchedulePolicy:
    """Cooling on (1) inside the windows - and, with a setpoint, only while the room is above it.

    Parameters
    ----------
    env : gym.Env
        The environment it will be evaluated in (may be wrapped).
    windows : list of dict or dict
        See schedule_available(). An empty list never cools.
    utc_offset_hours : float or str
        Local time = UTC + offset, or a time zone name such as "Europe/London" (see utc_offset).
    setpoint : float or None
        Switch mode only: cool only while the model's (first) room temperature is above it.
    """

    def __init__(self, env, windows, utc_offset_hours=0.0, setpoint=None):
        self.base_env = env.unwrapped
        self.windows = normalise_windows(windows)
        self.utc_offset_hours = utc_offset(utc_offset_hours)
        self.setpoint = None if setpoint is None else float(setpoint)

    def available(self, time):
        return schedule_available(time, self.windows, self.utc_offset_hours)

    def compute_single_action(self, obs=None, explore=None):
        row = self.base_env.observation[-1]
        on = bool(self.available(row[0].item()))
        if on and self.setpoint is not None:
            on = row[3].item() > self.setpoint  # row = [time, T1, T2, room, ...]
        return int(on)

    def __repr__(self):
        windows = ", ".join(f"{w['weekdays']} {w['start']:g}-{w['end']:g}h" for w in self.windows) or "never"
        setpoint = "" if self.setpoint is None else f", setpoint={self.setpoint:g}"
        return f"SchedulePolicy({windows}{setpoint})"
