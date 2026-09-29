"""
Irradiance on a building's external walls, from global horizontal irradiance alone.

rcmodel's envelope is driven by a sol-air temperature, T_sa = T_out + k_sa * I / H_OUT (ASHRAE Handbook -
Fundamentals, 2021, ch. 18, "sol-air temperature"). With only GHI available, I used to be GHI itself: a
horizontal quantity standing in for what falls on vertical walls, with the difference soaked up by k_sa.
This module works out I properly - the area-weighted irradiance on the building's actual external walls,
each facing its own way - from the same GHI, the site's location and the time. It uses pvlib (Holmgren,
Hansen & Mikofski 2018, JOSS 3(29):884; Anderson et al. 2023, JOSS 8(92):5994) rather than
re-implementing the solar geometry, so every step is a published, maintained model:

1. Sun position: NREL's Solar Position Algorithm (Reda & Andreas 2004, Solar Energy 76(5):577-589;
   corrigendum 2007, Solar Energy 81(6):838) - pvlib.solarposition.get_solarposition, method "nrel_numpy".
2. GHI split into direct normal (DNI) and diffuse horizontal (DHI): the Erbs correlation (Erbs, Klein &
   Duffie 1982, Solar Energy 28(4):293-302) - pvlib.irradiance.erbs. Erbs was fitted on hourly data;
   applied to finer data it is used as-is.
3. Irradiance on each vertical wall: beam + sky diffuse + ground-reflected - pvlib.irradiance.
   get_total_irradiance. The sky diffuse model defaults to Perez (Perez et al. 1990, Solar Energy
   44(5):271-289), the anisotropic model family EnergyPlus uses; "isotropic" (Loutzenhiser et al. 2007,
   Solar Energy 81(2):254-267) is available for comparison.

pvlib is imported only when this is used, so a model without solar_geometry never needs it.
"""

import numpy as np
import pandas as pd

SKY_MODELS = ("perez", "isotropic", "haydavies", "reindl", "klucher")


def envelope_irradiance(time, ghi, wall_azimuths, wall_areas, latitude, longitude, albedo=0.2, sky_model="perez"):
    """
    Area-weighted irradiance on the external walls, W/m2, at each time.

    Parameters
    ----------
    time : array of float        unix epoch seconds, UTC.
    ghi : array of float         global horizontal irradiance at `time`, W/m2.
    wall_azimuths : array        outward azimuth of each external wall, degrees clockwise from north.
    wall_areas : array           area of each external wall, m2 (the weights).
    latitude, longitude : float  site, degrees (north and east positive).
    albedo : float               ground reflectance for the ground-reflected component.
    sky_model : str              pvlib sky diffuse model, one of SKY_MODELS.

    Returns
    -------
    np.ndarray of float64, len(time): the area-weighted plane-of-array global irradiance. Night and any
    time pvlib cannot resolve (sun below the horizon) give 0.
    """
    import pvlib

    if sky_model not in SKY_MODELS:
        raise ValueError(f"sky_model must be one of {SKY_MODELS}, got {sky_model!r}.")
    time = np.asarray(time, dtype=np.float64)
    ghi = pd.Series(np.asarray(ghi, dtype=np.float64), index=pd.to_datetime(time, unit="s", utc=True))
    wall_azimuths = np.asarray(wall_azimuths, dtype=np.float64)
    wall_areas = np.asarray(wall_areas, dtype=np.float64)
    if wall_azimuths.shape != wall_areas.shape or wall_areas.sum() <= 0:
        raise ValueError("Need one azimuth per external wall and a positive total wall area.")

    index = ghi.index
    sun = pvlib.solarposition.get_solarposition(index, latitude, longitude, method="nrel_numpy")
    split = pvlib.irradiance.erbs(ghi, sun["zenith"], index)
    extra = {}
    if sky_model in ("perez", "haydavies", "reindl"):
        extra["dni_extra"] = pvlib.irradiance.get_extra_radiation(index)
    if sky_model == "perez":
        extra["airmass"] = pvlib.atmosphere.get_relative_airmass(sun["apparent_zenith"])

    total = np.zeros(len(index))
    for azimuth, area in zip(wall_azimuths, wall_areas, strict=True):
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=90.0,
            surface_azimuth=azimuth,
            solar_zenith=sun["apparent_zenith"],
            solar_azimuth=sun["azimuth"],
            dni=split["dni"],
            ghi=ghi,
            dhi=split["dhi"],
            albedo=albedo,
            model=sky_model,
            **extra,
        )["poa_global"]
        total += area * np.nan_to_num(np.asarray(poa, dtype=np.float64), nan=0.0).clip(min=0.0)
    return total / wall_areas.sum()


__all__ = ["SKY_MODELS", "envelope_irradiance"]
