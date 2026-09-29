"""Irradiance on the external walls (rcmodel.physical.solar) and its use in the sol-air temperature."""

import numpy as np
import pytest
import torch

from rcmodel import model_creator
from rcmodel.physical import Building, Room
from rcmodel.physical.solar import envelope_irradiance
from rcmodel.tools.helper_functions import change_origin

# EnergyPlus TrialA layouts (Data/EnergyPlus/CHANGES.md in using_rcmodel): +y is north, North Axis 0.
CORE = [[26.8, 11.6], [26.8, 3.7], [3.7, 3.7], [3.7, 11.6]]
FIVE_ZONE = [
    [[26.8, 3.7], [30.5, 0.0], [0.0, 0.0], [3.7, 3.7]],
    [[26.8, 11.6], [30.5, 15.2], [30.5, 0.0], [26.8, 3.7]],
    [[26.8, 11.6], [3.7, 11.6], [0.0, 15.2], [30.5, 15.2]],
    [[3.7, 3.7], [0.0, 0.0], [0.0, 15.2], [3.7, 11.6]],
    CORE,
]
SITE = {"latitude": 51.2491, "longitude": -2.14466}


def _orientations(coords, north_axis=0.0):
    building = Building([Room(str(i), c) for i, c in enumerate(change_origin(coords))], 2.4)
    azimuths, areas = building.external_wall_orientations(north_axis)
    return dict(zip(np.round(azimuths).astype(int).tolist(), np.round(areas, 2).tolist(), strict=True))


def test_external_wall_orientations_one_zone_box():
    assert _orientations([CORE]) == {0: 55.44, 90: 18.96, 180: 55.44, 270: 18.96}


def test_external_wall_orientations_five_zone_perimeter():
    """Only the four perimeter facades are external, and each faces out of its own room."""
    assert _orientations(FIVE_ZONE) == {0: 73.2, 90: 36.48, 180: 73.2, 270: 36.48}


def test_north_axis_rotates_every_wall():
    assert _orientations([CORE], north_axis=30.0) == {30: 55.44, 120: 18.96, 210: 55.44, 300: 18.96}


def _summer_day():
    """One clear-ish mid-June day at 10-minute steps, UTC, with a smooth GHI."""
    t = 1_686_787_200 + np.arange(0, 86_400, 600)  # 2023-06-15 00:00 UTC
    hours = (t - t[0]) / 3600
    ghi = np.clip(850 * np.sin(np.pi * (hours - 4.0) / 16.5), 0, None)
    return t, ghi


def test_walls_get_no_irradiance_at_night():
    t, ghi = _summer_day()
    irradiance = envelope_irradiance(t, ghi, [180.0], [1.0], **SITE)
    assert np.all(irradiance[ghi == 0] == 0)
    assert np.all(np.isfinite(irradiance)) and np.all(irradiance >= 0)


def test_south_wall_beats_north_wall_at_noon_and_east_leads_west():
    t, ghi = _summer_day()
    walls = {az: envelope_irradiance(t, ghi, [az], [1.0], **SITE) for az in (0.0, 90.0, 180.0, 270.0)}
    noon = np.argmin(np.abs((t - t[0]) / 3600 - 12.1))  # solar noon ~12:08 UTC at 2.1 W
    assert walls[180.0][noon] > 2 * walls[0.0][noon]
    assert np.argmax(walls[90.0]) < noon < np.argmax(walls[270.0])


def test_envelope_irradiance_is_area_weighted():
    t, ghi = _summer_day()
    south = envelope_irradiance(t, ghi, [180.0], [1.0], **SITE)
    east = envelope_irradiance(t, ghi, [90.0], [1.0], **SITE)
    both = envelope_irradiance(t, ghi, [180.0, 90.0], [3.0, 1.0], **SITE)
    np.testing.assert_allclose(both, (3 * south + east) / 4, rtol=1e-12)


@pytest.mark.parametrize("sky_model", ["perez", "isotropic"])
def test_sky_models_run(sky_model):
    t, ghi = _summer_day()
    assert envelope_irradiance(t, ghi, [180.0], [1.0], sky_model=sky_model, **SITE).max() > 100


def test_sol_air_uses_the_wall_irradiance_with_solar_geometry(get_model_config, fake_time, fake_ghi):
    """With solar_geometry the envelope sees the irradiance on its walls, not GHI - and the room's own
    solar gain term still uses GHI."""
    config = {**get_model_config, "solar_geometry": SITE}
    model = model_creator(config)
    model.setup()
    t = torch.tensor(fake_time, dtype=torch.float64)
    walls = model.envelope_irradiance(t)
    np.testing.assert_allclose(model.ghi(t), fake_ghi)
    assert not np.allclose(walls, fake_ghi)

    outdoor = np.asarray(model.Tout_continuous(t), dtype=np.float64).flatten()
    np.testing.assert_allclose(model.sol_air_temperature(t), outdoor + model.k_sa * walls / 25.0, rtol=1e-9)

    plain = model_creator(get_model_config)
    plain.setup()
    np.testing.assert_allclose(plain.envelope_irradiance(t), fake_ghi)


def test_solar_geometry_config_is_checked(get_model_config):
    with pytest.raises(AssertionError, match="longitude"):
        model_creator({**get_model_config, "solar_geometry": {"latitude": 51.0}})
    with pytest.raises(AssertionError, match="unknown"):
        model_creator({**get_model_config, "solar_geometry": {**SITE, "tilt": 90}})
