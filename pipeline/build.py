"""Build the wind atlas: HRRR model levels -> AGL ladder -> PNG tiles.

Run: PYTHONPATH=pipeline .venv/bin/python pipeline/build.py --out site/data \
       --leads 3 --workers 11
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from agl_interp import (
    apply_surface_rung,
    build_ladder_index,
    interp_field,
    omega_to_w,
)
from config import (AGL_LADDER, CP, G, R_DRY, N_BRUNT_SCALE, N_MODEL_LEVELS, OUT_NX, OUT_NY,
                    PBLH_SCALE, SOAR_SCALES, STAB_TOP_M, TILE)
import soaring
from hrrr_source import open_groups, pick_init
from reproject import build_index_map, regrid, rotate_winds
import tile_writer

log = logging.getLogger("build")

# Fields read per lead. Reads are network-latency bound, so they are issued
# concurrently; measured 167 s/lead at 6 workers against 79 s at 11.
LEVEL_VARS = ["wind_u", "wind_v", "geopotential_height", "vertical_velocity",
              "pressure", "temperature"]
SURFACE_VARS = ["geopotential_height_surface", "wind_u_10m", "wind_v_10m",
                # Boundary-layer depth, for the solver's stability knob. HRRR
                # ships it directly, so no derivation is needed — and it is a
                # real diurnal signal: measured 509 m at 15Z over Bozeman
                # against 2623 m in the afternoon and 259 m at night.
                "planetary_boundary_layer_height_surface",
                # The soaring forecast (pipeline/soaring.py). All surface
                # fields, so they cost one fetch each and no extra model-level
                # reads — the expensive part of a lead.
                "sensible_heat_flux_surface",   # the thermal engine
                "wind_gust_surface",            # gust factor = launch safety
                "temperature_2m",
                "pressure_surface",
                "pressure_cloud_base",          # NaN where there is no cloud
                "total_cloud_cover_boundary_layer"]


def read_lead(ml, sfc, init, lead, workers):
    sub = ml.sel(init_time=init, lead_time=lead).isel(
        model_level=slice(0, N_MODEL_LEVELS))
    s = sfc.sel(init_time=init, lead_time=lead)
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(lambda n=n: getattr(sub, n).values): n for n in LEVEL_VARS}
        futures.update({ex.submit(lambda n=n: getattr(s, n).values): n for n in SURFACE_VARS})
        for fut, name in futures.items():
            out[name] = fut.result()
    return out


def build_lead(raw, index_map):
    """Native model levels -> earth-relative wind on the output grid."""
    agl = raw["geopotential_height"] - raw["geopotential_height_surface"][:, :, None]
    idx, frac, shape = build_ladder_index(agl)

    u = interp_field(raw["wind_u"], idx, frac, shape)
    v = interp_field(raw["wind_v"], idx, frac, shape)
    w_ms = omega_to_w(raw["vertical_velocity"], raw["pressure"], raw["temperature"])
    w = interp_field(w_ms, idx, frac, shape)

    # The 10 m rung is below HRRR's lowest model level in every column, so it
    # comes from the model's own diagnostic rather than an extrapolation.
    u, v = apply_surface_rung(u, v, raw["wind_u_10m"], raw["wind_v_10m"])

    nk = len(AGL_LADDER)
    ug = np.empty((OUT_NY, OUT_NX, nk), dtype=np.float32)
    vg = np.empty_like(ug)
    wg = np.empty_like(ug)
    for k in range(nk):
        ug[:, :, k] = regrid(u[:, :, k], index_map)
        vg[:, :, k] = regrid(v[:, :, k], index_map)
        wg[:, :, k] = regrid(w[:, :, k], index_map)

    # HRRR's winds are grid-relative; the viewer works in earth coordinates.
    for k in range(nk):
        ug[:, :, k], vg[:, :, k] = rotate_winds(ug[:, :, k], vg[:, :, k], index_map)

    # Stability, for the solver's alpha knob. Both fields are smooth and
    # synoptic-scale, so they ride the same regrid as the wind.
    pblh = regrid(raw["planetary_boundary_layer_height_surface"], index_map)
    n_brunt = regrid(bulk_brunt_vaisala(raw, agl), index_map)

    # The soaring forecast. Regridded first, then derived, so every quantity is
    # computed on the grid the viewer reads rather than interpolated after the
    # fact — w* is a cube root and B/S a cube, so interpolating the OUTPUTS
    # would not equal deriving from interpolated inputs.
    wind10 = np.hypot(regrid(raw["wind_u_10m"], index_map),
                      regrid(raw["wind_v_10m"], index_map))
    soar = soaring.derive({
        "heat_flux": regrid(raw["sensible_heat_flux_surface"], index_map),
        "pbl_height": pblh,
        "temp_c": regrid(raw["temperature_2m"], index_map),
        "wind_10m": wind10,
        "gust": regrid(raw["wind_gust_surface"], index_map),
        "pressure_surface": regrid(raw["pressure_surface"], index_map),
        "pressure_cloud_base": regrid(raw["pressure_cloud_base"], index_map),
        "cloud_cover": regrid(raw["total_cloud_cover_boundary_layer"], index_map),
    })
    return ug, vg, wg, pblh, n_brunt, soar


def bulk_brunt_vaisala(raw, agl):
    """N over the layer the solver covers, in s^-1, on the NATIVE grid.

    One bulk value per column rather than a profile: the browser applies a
    single alpha to a ~50 km window, so vertical detail here would only be
    averaged away.

    Potential temperature needs pressure and temperature, both of which the
    build already reads for the omega->w conversion, so this costs one extra
    surface field and no extra reads of the expensive model-level data.

    Convective layers have dtheta/dz <= 0, where N is undefined; the floor
    turns those into a very small N, which the Froude mapping reads as
    "neutral" — the right answer, since convective air has no trouble climbing.
    """
    t_k = raw["temperature"] + 273.15
    theta = t_k * (100000.0 / raw["pressure"]) ** (R_DRY / CP)
    # Bracket the layer: level 0 is ~10 m AGL, and STAB_TOP_M is the depth the
    # solver actually resolves.
    top = np.argmin(np.abs(agl - STAB_TOP_M), axis=2)
    ii, jj = np.indices(top.shape)
    th_lo, th_hi = theta[:, :, 0], theta[ii, jj, top]
    z_lo, z_hi = agl[:, :, 0], agl[ii, jj, top]
    dz = np.maximum(z_hi - z_lo, 1.0)
    n2 = (G / th_lo) * (th_hi - th_lo) / dz
    return np.sqrt(np.maximum(n2, 1e-6)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--leads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--init", default="auto")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    sfc, ml = open_groups()
    init = pick_init(ml, sfc, args.leads, args.init)
    lead_values = ml.lead_time.values[: args.leads]
    log.info("init %s, %d leads", init, len(lead_values))

    index_map = build_index_map(ml.x.values, ml.y.values)
    valid = index_map["valid"]
    log.info("index map built (%.1f%% of the output grid is in-domain)",
             100 * valid.mean())

    os.makedirs(args.out, exist_ok=True)
    tiles_dir = os.path.join(args.out, "tiles")

    # Pass 1: read and interpolate every lead, SPILLING EACH TO DISK.
    #
    # The quantisation scales must be global — per-lead scales make the wind's
    # brightness pulse as the animation advances — but that does not require
    # every lead in memory at once, and holding them is what killed the build
    # at 18 leads. One lead is 0.53 GB (2048x1280x17, three float32 fields), so
    # 13 leads is 7 GB, and np.stack then COPIES it all to compute the scales.
    # The runner has 16 GB and the OOM killer took it at lead 13, twice,
    # identically — reads were healthy at 26 s throughout, which is what ruled
    # out a hang.
    #
    # Each lead is now reduced to 3 x 17 percentiles as it is read and written
    # to a temporary .npz; peak memory is one lead rather than all of them.
    spill = tempfile.mkdtemp(prefix="windbuild-")
    per_lead = []
    names = []
    try:
        for i, lead in enumerate(lead_values):
            t = time.time()
            raw = read_lead(ml, sfc, init, lead, args.workers)
            ug, vg, wg, pblh, nb, soar = build_lead(raw, index_map)
            name = f"f{i:02d}"
            names.append(name)
            per_lead.append(tile_writer.lead_percentiles(ug, vg, wg))
            # UNCOMPRESSED on purpose. savez_compressed was measured at 29.6 s
            # per lead against 1.9 s, to save 0.09 GB — nine extra minutes of
            # build for disk that is not the binding constraint. Each spill
            # file is deleted the moment its tiles are written (see write_one),
            # so the peak is bounded by how far the writers lag the readers
            # rather than by the forecast length.
            np.savez(os.path.join(spill, f"{name}.npz"), ug=ug, vg=vg, wg=wg,
                     pblh=pblh, nb=nb,
                     **{f"soar_{k}": v for k, v in soar.items()})
            del raw, ug, vg, wg, pblh, nb, soar
            log.info("  lead %d/%d in %.0f s", i + 1, len(lead_values),
                     time.time() - t)

        scales = tile_writer.scales_from_percentiles(per_lead)
        log.info("scales u=%s", [round(s) for s in scales["u"]])

        # Writing dominates the build: at 12 leads and 17 rungs it was 24.5 of
        # the 29 minutes, against ~22 s per lead to read. Each lead is 41 PNGs
        # of 17 stacked levels, and PNG encoding is CPU-bound and independent
        # per lead, so it parallelises cleanly. Threads rather than processes
        # because the encode happens inside Pillow, which releases the GIL.
        stab_scales = {"pblh": PBLH_SCALE, "n": N_BRUNT_SCALE}

        def write_one(name):
            # Loaded and released per lead, so the write stage's memory is
            # bounded by the number of writer threads rather than by the
            # forecast length.
            with np.load(os.path.join(spill, f"{name}.npz")) as z:
                qu = tile_writer.quantise(z["ug"], scales["u"])
                qv = tile_writer.quantise(z["vg"], scales["v"])
                qw = tile_writer.quantise(z["wg"], scales["w"])
                soar = {k[5:]: z[k] for k in z.files if k.startswith("soar_")}
                n = (tile_writer.write_lead_tiles(tiles_dir, name, qu, qv, qw)
                     + tile_writer.write_overview(tiles_dir, name, qu, qv, qw)
                     + tile_writer.write_stability(tiles_dir, name, z["pblh"],
                                                   z["nb"], stab_scales)
                     + tile_writer.write_soaring(tiles_dir, name, soar,
                                                 SOAR_SCALES))
            os.remove(os.path.join(spill, f"{name}.npz"))
            return n

        total = 0
        # Fewer writers than readers: each holds a decompressed lead, so this
        # is the memory knob now.
        with ThreadPoolExecutor(max_workers=min(4, len(names))) as ex:
            for n in ex.map(write_one, names):
                total += n

        total += tile_writer.write_mask(args.out, valid)
    finally:
        shutil.rmtree(spill, ignore_errors=True)

    meta = {
        "init": str(init),
        "leads": names,
        # lead_time comes back as numpy timedelta64; convert via hours.
        "leadHours": [int(np.timedelta64(v, "h").astype(int)) for v in lead_values],
        "agl": AGL_LADDER,
        "grid": {"nx": OUT_NX, "ny": OUT_NY, "tile": TILE,
                 "tilesX": OUT_NX // TILE, "tilesY": OUT_NY // TILE},
        "bounds": {"west": -125.5, "east": -66.5, "south": 23.5, "north": 50.5},
        "scales": scales,
        "overviewFactor": 4,
        # Stability product. The viewer treats its absence as "no stability
        # data" and falls back to a neutral solve, so a site built before this
        # existed keeps working.
        "stab": {"scales": stab_scales, "topM": STAB_TOP_M},
        # The soaring product. Absent in older builds, and the viewer treats
        # that as "no flight planning available" rather than failing.
        "soar": {"scales": SOAR_SCALES, "factor": tile_writer.SOAR_FACTOR},
    }
    tile_writer.write_meta(args.out, meta)

    log.info("wrote %.1f MB in %.0f s", total / 1e6, time.time() - t0)


if __name__ == "__main__":
    main()
