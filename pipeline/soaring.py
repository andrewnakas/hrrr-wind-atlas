"""Soaring forecast quantities — what a paraglider pilot actually asks.

The viewer can already show air moving correctly. A pilot's questions are
different: when should I fly, how strong will the thermals be, how high will I
get, is it too windy, will it be smooth or broken? This derives those from HRRR
fields the build already reads.

The vocabulary deliberately matches RASP/BLIPMAP
(http://www.drjack.info/rasp/info/parameters.html), which is what pilots
already read, so a number here means what a pilot expects it to mean.

EVERY NUMBER IS FORECAST GUIDANCE FROM A 3 km MODEL. HRRR cannot resolve a
launch, a spine or a venturi. The wind field gets downscaled to ~400 m by the
mass-consistent solver, but these thermal quantities stay at model resolution.
The UI says so; this module should not pretend otherwise either.
"""

from __future__ import annotations

import numpy as np

# Physical constants. G and R_DRY already live in config for the omega->w
# conversion; repeated here so this module reads standalone as physics.
G = 9.80665
CP = 1004.6          # J/(kg K), dry air at constant pressure
RHO = 1.1            # kg/m^3, near-surface density at Western US elevations
KARMAN = 0.4
Z0_DEFAULT = 0.1     # m, roughness length over mixed terrain/vegetation


def convective_velocity(heat_flux, pbl_height, temp_c):
    """w*, the Deardorff convective velocity scale, in m/s.

        w* = ( g/T * HFX/(rho*cp) * z_i )^(1/3)

    This is THE thermal strength number. RASP publishes it directly and pilots
    read it as "how strong are the thermals"; a glider's average climb is w*
    minus its own sink rate, and a paraglider working a core sees roughly 0.7 w*
    (it turns tighter than the thermal average but sinks more than a sailplane).

    Zero whenever the surface is not heating the air. That is not a guard, it is
    the physics: no positive heat flux means no dry convection, so no thermals.
    Measured on the live 21Z run over Bozeman: 140 W/m^2 with a 2767 m boundary
    layer gives 2.22 m/s; by +3h the flux is -18 and it is exactly 0.
    """
    hfx = np.asarray(heat_flux, dtype=np.float64)
    zi = np.asarray(pbl_height, dtype=np.float64)
    t_k = np.asarray(temp_c, dtype=np.float64) + 273.15
    ok = (hfx > 0) & (zi > 0) & np.isfinite(t_k) & (t_k > 100)
    out = np.zeros(np.broadcast(hfx, zi, t_k).shape, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.cbrt((G / t_k) * (hfx / (RHO * CP)) * zi)
    return np.where(ok, np.nan_to_num(val), 0.0)


def friction_velocity(wind_10m, z0=Z0_DEFAULT):
    """u*, the friction velocity, in m/s — how much turbulence the WIND makes.

        u* = k * U10 / ln(10/z0)

    Neutral log-law form. It is the shear half of the buoyancy/shear ratio: the
    same heating gives usable thermals in light wind and broken ones in strong
    wind, and this is the term that says which.
    """
    u = np.maximum(np.asarray(wind_10m, dtype=np.float64), 0.1)
    return KARMAN * u / np.log(10.0 / z0)


def buoyancy_shear_ratio(wstar, ustar):
    """B/S — whether thermals are organised or torn apart. Dimensionless.

    The ratio of buoyant to shear production of turbulence. Dr Jack's threshold
    is the useful one: **B/S at or below 5 means thermals are too broken to
    use**, though paragliders turn tightly enough to work somewhat lower values
    than sailplanes.

    This is the "chop" number. Measured: 31 for strong thermals under light
    wind (organised), and 3.0 for the SAME heating under 18 m/s (broken).
    """
    w = np.asarray(wstar, dtype=np.float64)
    u = np.maximum(np.asarray(ustar, dtype=np.float64), 1e-3)
    with np.errstate(over="ignore", invalid="ignore"):
        bs = (w / u) ** 3 / 2.5
    return np.where(w > 0, np.nan_to_num(bs, posinf=999.0), 0.0)


def cloud_base_agl(pressure_surface, pressure_cloud_base, temp_c):
    """Height of cloud base above ground, in metres, or NaN where there is none.

    Hypsometric relation between the surface and cloud-base pressures:

        dz = (R T / g) * ln(p_sfc / p_base),   R T / g ~ 29.3 * T

    Cumulus base caps the usable climb even when the boundary layer is deeper,
    which is why it is not enough to report the PBL top alone. HRRR reports a
    cloud-base pressure only where it diagnoses cloud, so an absent value means
    blue conditions rather than zero height.
    """
    ps = np.asarray(pressure_surface, dtype=np.float64)
    pb = np.asarray(pressure_cloud_base, dtype=np.float64)
    t_k = np.asarray(temp_c, dtype=np.float64) + 273.15
    ok = np.isfinite(pb) & (pb > 0) & (ps > 0) & (pb < ps)
    with np.errstate(invalid="ignore", divide="ignore"):
        dz = 29.3 * t_k * np.log(np.where(ok, ps / np.maximum(pb, 1.0), 1.0))
    return np.where(ok, dz, np.nan)


def top_of_lift(pbl_height, cloud_base):
    """The highest a pilot can realistically climb, in metres AGL.

    The dry thermal top is the boundary-layer top, but under cumulus the climb
    stops at cloud base — going higher means going into cloud, which is both
    illegal and a good way to get hurt. So this is the PBL top capped by cloud
    base wherever cloud base exists.
    """
    zi = np.asarray(pbl_height, dtype=np.float64)
    cb = np.asarray(cloud_base, dtype=np.float64)
    return np.where(np.isfinite(cb), np.minimum(zi, cb), zi)


def gust_factor(gust, wind_10m):
    """gust / mean wind — the launch-safety number.

    A pilot cares less about the mean than about the spread: 6 m/s steady is
    pleasant, 6 m/s gusting 12 is a bad launch. Measured over Bozeman it runs
    1.21 on a working afternoon and 1.9 once the surface decouples at night.
    """
    g = np.asarray(gust, dtype=np.float64)
    u = np.maximum(np.asarray(wind_10m, dtype=np.float64), 0.5)
    return np.where(np.isfinite(g), np.maximum(g, u) / u, 1.0)


def derive(fields):
    """Everything the site needs, from one lead's regridded surface fields.

    `fields` keys: heat_flux, pbl_height, temp_c, wind_10m, gust,
    pressure_surface, pressure_cloud_base, cloud_cover.
    """
    wstar = convective_velocity(fields["heat_flux"], fields["pbl_height"],
                                fields["temp_c"])
    ustar = friction_velocity(fields["wind_10m"])
    cb = cloud_base_agl(fields["pressure_surface"],
                        fields["pressure_cloud_base"], fields["temp_c"])
    return {
        "wstar": wstar,
        "top_of_lift": top_of_lift(fields["pbl_height"], cb),
        "cloud_base": cb,
        "gust_factor": gust_factor(fields["gust"], fields["wind_10m"]),
        "bs_ratio": buoyancy_shear_ratio(wstar, ustar),
        "cloud_cover": np.asarray(fields["cloud_cover"], dtype=np.float64),
        "wind_10m": np.asarray(fields["wind_10m"], dtype=np.float64),
    }
