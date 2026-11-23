"""The init picker, against small synthetic stores.

The newest init in the dataset is often still being ingested. A build that
starts on one reads every lead and then fails at the end, so the picker has to
step back past any init that is missing a lead the build will read, in either
the model-level group or the surface group. These checks fail if it only
probes one group or one lead, which is the bug that let 2026-11-23 00Z through.

Run:  .venv/bin/python test/test_init_picker.py
"""

import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

import hrrr_source  # noqa: E402

FAILURES = []

INITS = np.array(["2026-11-22T22", "2026-11-22T23", "2026-11-23T00"], dtype="datetime64[ns]")
N_LEAD = 5   # the dataset's leads
LEADS = 4    # the leads a build reads
NY = NX = 4


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def stores():
    """A complete (sfc, ml) pair; callers punch NaN holes into it."""
    coords = {
        "init_time": INITS,
        "lead_time": np.arange(N_LEAD).astype("timedelta64[h]").astype("timedelta64[ns]"),
        "y": np.arange(NY),
        "x": np.arange(NX),
    }
    ml = xr.Dataset(
        {"wind_u": (("init_time", "lead_time", "model_level", "y", "x"),
                    np.ones((len(INITS), N_LEAD, 2, NY, NX), np.float32))},
        coords={**coords, "model_level": np.arange(2)},
    )
    sfc = xr.Dataset(
        {"planetary_boundary_layer_height_surface":
            (("init_time", "lead_time", "y", "x"),
             np.full((len(INITS), N_LEAD, NY, NX), 500.0, np.float32))},
        coords=coords,
    )
    return sfc, ml


def picked(sfc, ml, **kw):
    try:
        return str(hrrr_source.pick_init(ml, sfc, LEADS, **kw))[:13]
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


def main():
    hrrr_source.PROBE_WINDOW = ((1, 3), (1, 3))

    sfc, ml = stores()
    got = picked(sfc, ml)
    check("a complete newest init is used", got == "2026-11-23T00", got)

    sfc, ml = stores()
    sfc["planetary_boundary_layer_height_surface"][2, 1] = np.nan
    got = picked(sfc, ml)
    check("a surface-group lead still ingesting steps back an hour", got == "2026-11-22T23", got)

    sfc, ml = stores()
    ml["wind_u"][2, 2] = np.nan
    got = picked(sfc, ml)
    check("a model-level lead in the middle still ingesting steps back", got == "2026-11-22T23", got)

    sfc, ml = stores()
    ml["wind_u"][2, N_LEAD - 1] = np.nan
    sfc["planetary_boundary_layer_height_surface"][2, N_LEAD - 1] = np.nan
    got = picked(sfc, ml)
    check("a missing lead the build does not read is no reason to step back",
          got == "2026-11-23T00", got)

    sfc, ml = stores()
    ml["wind_u"][2, 0, 0, 1:3, 1:3] = np.nan
    got = picked(sfc, ml)
    check("the probe reads the lowest model level", got == "2026-11-22T23", got)

    sfc, ml = stores()
    ml["wind_u"][2, 0, 0, 0, 0] = np.nan
    got = picked(sfc, ml)
    check("a NaN outside the probe window is not read as missing", got == "2026-11-23T00", got)

    sfc, ml = stores()
    sfc["planetary_boundary_layer_height_surface"][:, 0] = np.nan
    got = picked(sfc, ml, max_steps_back=2)
    check("no complete init raises rather than building a broken atlas",
          got == "RuntimeError", got)

    sfc, ml = stores()
    sfc["planetary_boundary_layer_height_surface"][1, 3] = np.nan
    got = picked(sfc, ml, requested="2026-11-22T23")
    check("a requested init that is incomplete is refused", got == "ValueError", got)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
