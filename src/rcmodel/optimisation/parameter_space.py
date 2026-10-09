"""
Search parameterisations that sit in front of the physical parameters.

Envelope ratios (Underwood, "An improved lumped parameter method for building thermal modelling", Energy and
Buildings 79 (2014) 191-201, eqs 7-11, after Gouda et al. 2002): a 3R2C element is its total resistance and total
capacity, apportioned by rationing factors.

    R_o = f_o R_total    R_i = f_i R_total    R_m = (1 - f_o - f_i) R_total
    C_i = g_i C_total    C_o = (1 - g_i) C_total

In rcmodel's envelope R1 joins the outdoor air to the outer node C1 and R3 joins the inner node C2 to the room, so
R1 = R_o, R2 = R_m, R3 = R_i, C1 = C_o and C2 = C_i. R_total is m2K/W (like R1-R3) and C_total is J/K (like C1, C2).

Searching (R_total, C_total, f_o, f_i, g_i) instead of (R1, R2, R3, C1, C2) separates what the data identifies well
(the envelope's UA and its mass) from how they are distributed through the wall, so two candidates that differ only
in the split are seen to be close rather than far apart.

ParameterSpace maps a unit cube [0, 1]^d to physical values - continuous, or snapped to a log or linear grid - for an
optimiser that searches a box (pynmmso, scipy). The physics is unchanged: decode() hands back R1-R3, C1, C2.
"""

import math

import numpy as np

RATIO_KEYS = ("R_total", "C_total", "f_o", "f_i", "g_i")
ENVELOPE_PHYSICAL_KEYS = ("R1", "R2", "R3", "C1", "C2")


def envelope_from_ratios(R_total, C_total, f_o, f_i, g_i):
    """{R1, R2, R3, C1, C2} from the totals and rationing factors (see the module docstring).

    Raises ValueError unless every resistance and capacitance comes out > 0: R_total, C_total, f_o, f_i > 0,
    f_o + f_i < 1 and 0 < g_i < 1.
    """
    values = {"R_total": R_total, "C_total": C_total, "f_o": f_o, "f_i": f_i, "g_i": g_i}
    for key, value in values.items():
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{key} must be finite and > 0, got {value}")
    if not f_o + f_i < 1:
        raise ValueError(f"f_o + f_i must be < 1 so R2 > 0, got {f_o} + {f_i}")
    if not g_i < 1:
        raise ValueError(f"g_i must be < 1 so C1 > 0, got {g_i}")
    return {
        "R1": f_o * R_total,
        "R2": (1 - f_o - f_i) * R_total,
        "R3": f_i * R_total,
        "C1": (1 - g_i) * C_total,
        "C2": g_i * C_total,
    }


def ratios_from_envelope(physical):
    """{R_total, C_total, f_o, f_i, g_i} from a dict holding R1, R2, R3, C1, C2: the inverse of envelope_from_ratios."""
    r_total = physical["R1"] + physical["R2"] + physical["R3"]
    c_total = physical["C1"] + physical["C2"]
    return {
        "R_total": r_total,
        "C_total": c_total,
        "f_o": physical["R1"] / r_total,
        "f_i": physical["R3"] / r_total,
        "g_i": physical["C2"] / c_total,
    }


def envelope_ranges(spec):
    """The smallest [min, max] for each of R1-R3, C1, C2 that holds every envelope the ratio ranges in `spec` decode
    to. Written into a model_config, these keep physical_to_scaled's machine values in [0, 1]."""
    (r_lo, r_hi), (c_lo, c_hi) = spec["R_total"]["range"], spec["C_total"]["range"]
    (fo_lo, fo_hi), (fi_lo, fi_hi), (g_lo, g_hi) = (spec[k]["range"] for k in ("f_o", "f_i", "g_i"))
    return {
        "R1": [fo_lo * r_lo, fo_hi * r_hi],
        "R2": [(1 - fo_hi - fi_hi) * r_lo, (1 - fo_lo - fi_lo) * r_hi],
        "R3": [fi_lo * r_lo, fi_hi * r_hi],
        "C1": [(1 - g_hi) * c_lo, (1 - g_lo) * c_hi],
        "C2": [g_lo * c_lo, g_hi * c_hi],
    }


class ParameterSpace:
    """
    The unit cube [0, 1]^d <-> physical parameter values.

    spec    {key: {"range": [lo, hi], "log": bool, "per_decade": int, "step": float}}, one entry per searched
            dimension, in vector order. Only "range" is required.
            per_decade  a log grid: n = round(per_decade * log10(hi / lo)) + 1 points from lo to hi.
            step        a linear grid lo, lo + step, ..., hi.
            neither     continuous, uniform in log10 space if "log" else linearly.
            A grid dimension snaps u to the nearest of its n points: grid[round(u * (n - 1))].
    pinned  {key: value} returned by decode() as well (fixed parameters, which may include ratio keys).

    When the searched and pinned keys include the RATIO_KEYS, decode() replaces them with R1-R3, C1, C2
    (envelope_from_ratios); all five must be present between the two.
    """

    def __init__(self, spec, pinned=None):
        self.spec = {key: dict(entry) for key, entry in spec.items()}
        self.pinned = dict(pinned or {})
        self.keys = list(self.spec)
        self._grids = {}
        for key, entry in self.spec.items():
            lo, hi = (float(v) for v in entry["range"])
            if not lo < hi:
                raise ValueError(f"{key}: range must have min < max, got {entry['range']} (pin it instead)")
            if entry.get("per_decade") and entry.get("step"):
                raise ValueError(f"{key}: give per_decade or step, not both")
            log = bool(entry.get("log")) or bool(entry.get("per_decade"))
            if log and lo <= 0:
                raise ValueError(f"{key}: a log dimension needs min > 0, got {lo}")
            entry["range"], entry["log"] = [lo, hi], log
            if entry.get("per_decade"):
                n = round(entry["per_decade"] * math.log10(hi / lo)) + 1
                self._grids[key] = np.geomspace(lo, hi, max(n, 2))
            elif entry.get("step"):
                n = round((hi - lo) / entry["step"]) + 1
                if not math.isclose(lo + (n - 1) * entry["step"], hi, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"{key}: range {entry['range']} is not a whole number of steps of {entry['step']}")
                # Rounded so that a 0.01 grid gives 0.07, not 0.07000000000000001.
                self._grids[key] = np.round(lo + entry["step"] * np.arange(n), 12)
        uses_ratios = [k for k in RATIO_KEYS if k in self.spec or k in self.pinned]
        if uses_ratios and len(uses_ratios) != len(RATIO_KEYS):
            missing = [k for k in RATIO_KEYS if k not in uses_ratios]
            raise ValueError(f"the envelope ratios need all of {RATIO_KEYS}; missing {missing}")
        self.uses_ratios = bool(uses_ratios)

    @property
    def dim(self):
        return len(self.keys)

    def bounds(self):
        """(lower, upper) of the unit cube, in the form pynmmso's get_bounds() returns."""
        return [0.0] * self.dim, [1.0] * self.dim

    def grid(self, key):
        """The points a grid dimension can take, or None for a continuous one."""
        return None if key not in self._grids else self._grids[key].copy()

    def values(self, u):
        """{key: value} of the searched dimensions only (ratio keys left as they are)."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        if u.shape != (self.dim,):
            raise ValueError(f"expected a vector of length {self.dim}, got shape {u.shape}")
        out = {}
        for key, x in zip(self.keys, u, strict=True):
            lo, hi = self.spec[key]["range"]
            if key in self._grids:
                grid = self._grids[key]
                out[key] = float(grid[round(float(x) * (len(grid) - 1))])
            elif self.spec[key]["log"]:
                out[key] = float(10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo))))
            else:
                out[key] = float(lo + x * (hi - lo))
        return out

    def decode(self, u):
        """The physical parameter dict for unit-cube point u: pinned values, then the searched ones, with the ratio
        keys (if used) replaced by R1-R3, C1, C2."""
        physical = {**self.pinned, **self.values(u)}
        if self.uses_ratios:
            physical.update(envelope_from_ratios(*(physical.pop(k) for k in RATIO_KEYS)))
        return physical

    def encode(self, physical):
        """The unit-cube point for a physical parameter dict (R1-R3, C1, C2 or the ratio keys), snapped to the
        grid dimensions and clipped to the ranges. decode(encode(p)) is p on the grid."""
        values = dict(physical)
        if self.uses_ratios and not all(k in values for k in RATIO_KEYS):
            values.update(ratios_from_envelope(values))
        u = []
        for key in self.keys:
            lo, hi = self.spec[key]["range"]
            value = float(values[key])
            if key in self._grids:
                grid = self._grids[key]
                scale = np.log(grid) if self.spec[key]["log"] else grid
                target = math.log(max(value, lo)) if self.spec[key]["log"] else value
                u.append(int(np.argmin(np.abs(scale - target))) / (len(grid) - 1))
            elif self.spec[key]["log"]:
                u.append((math.log10(max(value, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
            else:
                u.append((value - lo) / (hi - lo))
        return np.clip(np.array(u, dtype=float), 0.0, 1.0)
