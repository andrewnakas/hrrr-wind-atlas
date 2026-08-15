"""Verify the AGL ladder interpolation against real HRRR data.

Run: PYTHONPATH=pipeline .venv/bin/python test/test_agl_interp.py <slab.npz>
The slab is produced by the Phase 0 spike (wind_u, wind_v, geopotential_height,
geopotential_height_surface, ... on model levels).
"""

import sys
import numpy as np

sys.path.insert(0, "pipeline")
from agl_interp import (  # noqa: E402
    apply_surface_rung,
    build_ladder_index,
    interp_field,
    rung_is_extrapolated,
)
from config import AGL_LADDER  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main(path):
    d = np.load(path)
    gh = d["geopotential_height"]
    gsfc = d["geopotential_height_surface"]
    u, v = d["wind_u"], d["wind_v"]
    agl = gh - gsfc[:, :, None]
    ladder = np.asarray(AGL_LADDER)

    print(f"slab {u.shape}, surface {np.nanmin(gsfc):.0f}..{np.nanmax(gsfc):.0f} m")

    idx, frac, shape = build_ladder_index(agl)

    # The property the whole design rests on: interpolating the AGL field itself
    # must return the ladder heights exactly, for every column, everywhere.
    got = interp_field(agl, idx, frac, shape)
    err = np.abs(got - ladder[None, None, :])
    extrap = rung_is_extrapolated(agl)
    print("\n== ladder heights are reproduced exactly (interpolated rungs) ==")
    print(f"   model level 1 spans {agl[:, :, 0].min():.2f}..{agl[:, :, 0].max():.2f} m AGL")
    for k, target in enumerate(ladder):
        if extrap[k]:
            # Below HRRR's lowest model level in every column; supplied by the
            # 10 m diagnostic instead of interpolation. See apply_surface_rung.
            print(f"  SKIP  rung {target:6.0f} m — outside model range, uses diagnostic")
            continue
        check(f"rung {target:6.0f} m", float(err[:, :, k].max()) < 0.05,
              f"max err {err[:, :, k].max():.4f} m")
    interp_err = max(
        (float(err[:, :, k].max()) for k in range(len(ladder)) if not extrap[k]),
        default=0.0,
    )
    check("all interpolated rungs < 5 cm error", interp_err < 0.05, f"max {interp_err:.4f} m")
    check("only the surface rung is extrapolated", sum(extrap) <= 1,
          f"extrapolated rungs: {[ladder[i] for i, e in enumerate(extrap) if e]}")

    # Interpolated winds must never leave the range of their bracketing levels.
    print("\n== interpolated winds stay within the source column ==")
    U = interp_field(u, idx, frac, shape)
    V = interp_field(v, idx, frac, shape)
    for name, src, out in (("u", u, U), ("v", v, V)):
        lo, hi = np.nanmin(src, axis=2), np.nanmax(src, axis=2)
        inside = ((out >= lo[:, :, None] - 1e-3) & (out <= hi[:, :, None] + 1e-3))
        check(f"{name} within column min/max", bool(inside.all()),
              f"{100*inside.mean():.4f}% inside")
        check(f"{name} finite", bool(np.isfinite(out).all()))

    # Independent physical cross-check, then the surface-rung substitution.
    if "wind_u_10m" in d:
        u10, v10 = d["wind_u_10m"], d["wind_v_10m"]
        diff = np.abs(np.hypot(U[:, :, 0], V[:, :, 0]) - np.hypot(u10, v10))
        print("\n== 10 m rung vs HRRR's diagnostic 10 m wind ==")
        check("extrapolated rung is already close", float(np.nanmedian(diff)) < 0.75,
              f"median {np.nanmedian(diff):.3f}, p95 {np.nanpercentile(diff,95):.3f} m/s")

        U, V = apply_surface_rung(U, V, u10, v10)
        exact = np.allclose(U[:, :, 0], u10) and np.allclose(V[:, :, 0], v10)
        check("surface rung now IS the diagnostic wind", exact)

        # Substituting the surface rung must not introduce a discontinuity with
        # the rung above it.
        shear = np.abs(np.hypot(U[:, :, 1], V[:, :, 1]) - np.hypot(U[:, :, 0], V[:, :, 0]))
        check("10->25 m shear stays physical", float(np.nanpercentile(shear, 99.9)) < 6.0,
              f"p99.9 {np.nanpercentile(shear, 99.9):.2f} m/s over 15 m")

    # Monotonic ladder targets must map to non-decreasing source indices.
    print("\n== bracket indices are ordered ==")
    check("idx non-decreasing across rungs", bool((np.diff(idx.astype(np.int32), axis=1) >= 0).all()))
    check("frac in [0,1]", bool(((frac >= 0) & (frac <= 1)).all()))

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
