"""Envelope ratios (Underwood 2014) and the unit-cube ParameterSpace an external optimiser searches."""

import math

import numpy as np
import pytest

from rcmodel import (
    ENVELOPE_PHYSICAL_KEYS,
    ParameterSpace,
    envelope_from_ratios,
    envelope_ranges,
    model_creator,
    physical_to_scaled,
    ratios_from_envelope,
)

# The RC envelope EnergyPlus TrialA_1Zone's IDF implies (using_rcmodel/eplus_idf.py).
IDF_ENVELOPE = {"R1": 0.081, "R2": 1.330, "R3": 0.290, "C1": 2.895e7, "C2": 1.149e7}

SPEC = {
    "R_total": {"range": [0.2, 10.0], "per_decade": 20},
    "C_total": {"range": [1e6, 2e8], "per_decade": 20},
    "f_o": {"range": [0.01, 0.49], "step": 0.01},
    "f_i": {"range": [0.01, 0.49], "step": 0.01},
    "g_i": {"range": [0.01, 0.99], "step": 0.01},
    "k_sa": {"range": [0.0, 1.0]},
    "C_rm": {"range": [1e3, 1e5], "log": True},
}


def _random_points(space, n=200, seed=0):
    return np.random.default_rng(seed).random((n, space.dim))


# --------------------------------------------------------------------------- envelope <-> ratios


def test_ratio_round_trip_preserves_the_totals():
    ratios = ratios_from_envelope(IDF_ENVELOPE)
    assert ratios["R_total"] == pytest.approx(0.081 + 1.330 + 0.290)
    assert ratios["C_total"] == pytest.approx(2.895e7 + 1.149e7)
    assert ratios["f_o"] == pytest.approx(0.081 / 1.701)
    assert ratios["f_i"] == pytest.approx(0.290 / 1.701)
    assert ratios["g_i"] == pytest.approx(1.149e7 / 4.044e7)

    back = envelope_from_ratios(**ratios)
    for key in ENVELOPE_PHYSICAL_KEYS:
        assert back[key] == pytest.approx(IDF_ENVELOPE[key])


def test_ratios_split_as_in_the_paper():
    envelope = envelope_from_ratios(R_total=2.0, C_total=1e7, f_o=0.1, f_i=0.25, g_i=0.3)
    assert envelope == pytest.approx({"R1": 0.2, "R2": 1.3, "R3": 0.5, "C1": 7e6, "C2": 3e6})


@pytest.mark.parametrize(
    "ratios",
    [
        {"f_o": 0.5, "f_i": 0.5, "g_i": 0.5},  # R2 = 0
        {"f_o": 0.6, "f_i": 0.5, "g_i": 0.5},  # R2 < 0
        {"f_o": 0.1, "f_i": 0.1, "g_i": 1.0},  # C1 = 0
        {"f_o": 0.1, "f_i": 0.1, "g_i": 0.0},  # C2 = 0
        {"f_o": 0.0, "f_i": 0.1, "g_i": 0.5},  # R1 = 0
        {"f_o": 0.1, "f_i": float("nan"), "g_i": 0.5},
    ],
)
def test_invalid_ratios_raise(ratios):
    with pytest.raises(ValueError):
        envelope_from_ratios(R_total=1.0, C_total=1e7, **ratios)


@pytest.mark.parametrize("totals", [{"R_total": 0.0, "C_total": 1e7}, {"R_total": 1.0, "C_total": -1.0}])
def test_non_positive_totals_raise(totals):
    with pytest.raises(ValueError):
        envelope_from_ratios(f_o=0.1, f_i=0.1, g_i=0.5, **totals)


# --------------------------------------------------------------------------- grids


def test_log_grid_spans_the_range_in_equal_ratios():
    space = ParameterSpace(SPEC)
    grid = space.grid("R_total")
    assert len(grid) == round(20 * math.log10(10 / 0.2)) + 1  # 35
    assert grid[0] == pytest.approx(0.2) and grid[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(grid[1:] / grid[:-1], (10 / 0.2) ** (1 / (len(grid) - 1)))
    assert space.grid("k_sa") is None


def test_step_grid_gives_whole_steps():
    space = ParameterSpace(SPEC)
    grid = space.grid("f_o")
    assert len(grid) == 49
    assert grid[0] == 0.01 and grid[-1] == 0.49
    assert all(round(v * 100) == pytest.approx(v * 100, abs=1e-9) for v in grid)


def test_step_must_divide_the_range():
    with pytest.raises(ValueError):
        ParameterSpace({"f_o": {"range": [0.01, 0.495], "step": 0.01}})


def test_decode_snaps_to_the_grid_and_ends_map_to_the_range():
    space = ParameterSpace(SPEC)
    grids = {key: space.grid(key) for key in ("R_total", "C_total", "f_o", "f_i", "g_i")}
    for u in _random_points(space):
        values = space.values(u)
        for key, grid in grids.items():
            assert np.min(np.abs(grid - values[key])) <= 1e-12 * max(1.0, abs(values[key]))
    lo, hi = space.values(np.zeros(space.dim)), space.values(np.ones(space.dim))
    for key, entry in SPEC.items():
        assert lo[key] == pytest.approx(entry["range"][0])
        assert hi[key] == pytest.approx(entry["range"][1])
    assert space.values(np.full(space.dim, 0.5))["C_rm"] == pytest.approx(1e4)  # log midpoint


def test_decode_returns_the_physical_envelope_and_the_pins():
    space = ParameterSpace(SPEC, pinned={"T_set": 22.0})
    physical = space.decode(np.full(space.dim, 0.3))
    assert set(ENVELOPE_PHYSICAL_KEYS) <= set(physical)
    assert not set(physical) & {"R_total", "C_total", "f_o", "f_i", "g_i"}
    assert physical["T_set"] == 22.0
    values = space.values(np.full(space.dim, 0.3))
    assert sum(physical[k] for k in ("R1", "R2", "R3")) == pytest.approx(values["R_total"])


def test_encode_decode_round_trip():
    space = ParameterSpace(SPEC)
    for u in _random_points(space, n=50, seed=1):
        physical = space.decode(u)
        again = space.decode(space.encode(physical))
        for key, value in physical.items():
            assert again[key] == pytest.approx(value, rel=1e-9)


def test_encode_snaps_an_off_grid_envelope():
    space = ParameterSpace(SPEC)
    snapped = ratios_from_envelope(space.decode(space.encode({**IDF_ENVELOPE, "k_sa": 0.5, "C_rm": 1e4})))
    assert snapped["f_o"] == pytest.approx(0.05)  # 0.0476
    assert snapped["f_i"] == pytest.approx(0.17)  # 0.1705
    assert snapped["g_i"] == pytest.approx(0.28)  # 0.2841
    assert snapped["R_total"] == pytest.approx(1.701, rel=0.07)  # within half a 12 % step


def test_ratios_must_all_be_given():
    with pytest.raises(ValueError, match="missing"):
        ParameterSpace({"R_total": SPEC["R_total"], "f_o": SPEC["f_o"]})
    # Pinned ratio keys count.
    space = ParameterSpace({"R_total": SPEC["R_total"]}, pinned={"C_total": 1e7, "f_o": 0.1, "f_i": 0.2, "g_i": 0.5})
    assert space.decode([0.0])["R1"] == pytest.approx(0.1 * 0.2)


# --------------------------------------------------------------------------- feeding the model


def test_envelope_ranges_hold_every_decoded_envelope():
    space = ParameterSpace(SPEC)
    ranges = envelope_ranges(SPEC)
    for u in _random_points(space):
        physical = space.decode(u)
        for key, (lo, hi) in ranges.items():
            assert lo * (1 - 1e-9) <= physical[key] <= hi * (1 + 1e-9)


def test_decoded_parameters_build_a_valid_model(get_model_config):
    spec = {key: entry for key, entry in SPEC.items() if key != "C_rm"}
    model_config = {**get_model_config, **envelope_ranges(spec)}
    pinned = {"C_rm": 1e4, "Rin": 1.0, "cool": 10.0, "gain": 1.0, "solar": 0.05}
    space = ParameterSpace(spec, pinned=pinned)
    for u in _random_points(space, n=3, seed=2):
        scaled = physical_to_scaled(model_config, space.decode(u))
        for key in ENVELOPE_PHYSICAL_KEYS:
            assert -1e-9 <= scaled[key] <= 1 + 1e-9
        model = model_creator({**model_config, "parameters": scaled})
        model.setup()
        physical = model.get_physical_paramaters()
        assert np.all(np.isfinite(np.asarray(physical[0], dtype=float)))
