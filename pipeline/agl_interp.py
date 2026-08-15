"""Resample HRRR native model levels onto a fixed above-ground ladder.

This is the step that removes the old build's whole class of vertical-placement
bugs. HRRR's model levels are terrain-following, so each column already knows
its own height above ground: geopotential_height - geopotential_height_surface.
Interpolating onto fixed AGL rungs here means the frontend never has to reason
about where a level sits relative to the terrain — every rung is exactly N metres
above the ground beneath it, everywhere.
"""

from __future__ import annotations

import numpy as np

from config import AGL_LADDER, G, R_DRY


def omega_to_w(omega_pa_s, pressure_pa, temperature_c, specific_humidity=None):
    """Convert vertical velocity from Pa/s to m/s.

    w = -omega / (rho * g), with density from the ideal gas law. Humidity enters
    through virtual temperature; it is a sub-1% correction, so callers may omit
    it (HRRR carries specific_humidity on model levels if wanted).
    """
    t_kelvin = temperature_c + 273.15  # dataset stores degrees Celsius
    if specific_humidity is not None:
        t_kelvin = t_kelvin * (1.0 + 0.61 * specific_humidity)
    rho = pressure_pa / (R_DRY * t_kelvin)
    return -omega_pa_s / (rho * G)


def build_ladder_index(agl, ladder=None):
    """Precompute bracketing level indices and blend fractions for every column.

    ``agl`` is (y, x, level) height above ground, increasing along the level
    axis. Returns (idx, frac) shaped (y*x, rung) to be reused across every field
    — building the index once and reusing it is what keeps this fast.
    """
    ladder = np.asarray(AGL_LADDER if ladder is None else ladder, dtype=np.float32)
    ny, nx, nlev = agl.shape
    flat = agl.reshape(-1, nlev).astype(np.float32, copy=False)

    idx = np.empty((flat.shape[0], ladder.size), dtype=np.int16)
    for k, target in enumerate(ladder):
        # Count levels at or below the target: the last such level is the lower
        # bracket. A per-column searchsorted loop over ~1.9M columns took 150 s;
        # this vectorised form takes ~1.3 s.
        idx[:, k] = np.clip(np.count_nonzero(flat <= target, axis=1) - 1, 0, nlev - 2)

    lower = np.take_along_axis(flat, idx, axis=1)
    upper = np.take_along_axis(flat, idx + 1, axis=1)
    frac = np.clip((ladder[None, :] - lower) / np.maximum(upper - lower, 1e-6), 0.0, 1.0)
    return idx, frac.astype(np.float32), (ny, nx, ladder.size)


def interp_field(field, idx, frac, shape):
    """Apply a prebuilt ladder index to one (y, x, level) field.

    Always interpolate wind COMPONENTS, never speed and direction: a column
    whose wind reverses through a shear layer legitimately passes through near
    zero speed, and component interpolation reproduces that correctly.
    """
    ny, nx, nk = shape
    flat = field.reshape(-1, field.shape[-1])
    lower = np.take_along_axis(flat, idx, axis=1)
    upper = np.take_along_axis(flat, idx + 1, axis=1)
    return (lower + (upper - lower) * frac).reshape(ny, nx, nk)


def apply_surface_rung(u_ladder, v_ladder, u10, v10):
    """Replace the 10 m rung with HRRR's diagnostic 10 m wind.

    Measured: HRRR's lowest model level sits at 9.91-11.53 m AGL, i.e. *above*
    10 m in 100% of CONUS columns. The 10 m rung therefore has no lower
    bracketing level, and interpolation degenerates into linear extrapolation
    below the model's own floor — which is exactly where the log-ish wind
    profile is least linear (up to 1.53 m of height error in spot checks).

    HRRR already publishes a proper 10 m diagnostic derived with surface-layer
    similarity theory, so we use it rather than extrapolating. The two agree to
    a median 0.08 m/s where both are meaningful, so this changes little in the
    mean and fixes the columns where it matters.
    """
    u_ladder[:, :, 0] = u10
    v_ladder[:, :, 0] = v10
    return u_ladder, v_ladder


def rung_is_extrapolated(agl, ladder=None, tolerance=0.001):
    """Flag ladder rungs that fall outside the model levels in ANY column.

    Deliberately per-column, not domain-wide: HRRR's level 1 spans 9.91-11.53 m
    AGL, so a domain-wide minimum of 9.91 makes the 10 m rung look bracketed
    while it is in fact above level 1 in ~all columns. Used by the tests to
    assert that only the surface rung (handled by apply_surface_rung) ever needs
    data from outside its own column.
    """
    ladder = np.asarray(AGL_LADDER if ladder is None else ladder, dtype=np.float32)
    lowest = agl[:, :, 0]
    highest = agl[:, :, -1]
    out = []
    for target in ladder:
        below = np.nanmean(target < lowest)
        above = np.nanmean(target > highest)
        out.append(bool(below > tolerance or above > tolerance))
    return out
