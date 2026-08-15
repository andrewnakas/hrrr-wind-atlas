"""The soaring forecast's physics, pinned by tests.

These numbers decide whether someone flies, so each one is checked against a
value measured from the live HRRR store rather than against itself. Several
checks exist specifically to FAIL if the physics is wired up plausibly but
wrongly — a thermal forecast that says "good" on a stable night is worse than
no forecast at all.

Run:  .venv/bin/python test/test_soaring.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

import soaring as sg  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# Measured from the live 21Z run over SW Montana, 2026-08-12. These are the
# anchors: if the physics stops reproducing them, something has drifted.
LIVE = [
    # lead, PBLH, HFX, T2m, gust, wind10
    (0, 2767, 140, 39.0, 7.3, 6.0),
    (3, 2667, -18, 36.5, 8.3, 5.2),
    (6, 270, -46, 30.9, 9.8, 5.1),
]
# The same rows without the lead index, for tests that do not care about it.
LIVE_ROWS = [r[1:] for r in LIVE]


def thermal_strength():
    print("\n== w*: thermal strength, against the live afternoon ==")
    zi, hfx, t = 2767, 140, 39.0
    w = float(sg.convective_velocity(hfx, zi, t))
    check("the live 21Z afternoon gives a workable thermal day",
          2.0 < w < 2.5, f"w* {w:.2f} m/s (measured 2.22)")
    # A paraglider core is ~0.7 w*, which should land in the range pilots
    # actually report on a day like this.
    check("...which is a plausible paraglider climb", 1.2 < 0.7 * w < 2.0,
          f"core ~{0.7 * w:.1f} m/s")

    # THE property that matters most for safety: no heating, no thermals.
    for lead, zi, hfx, t, _g, _u in LIVE:
        w = float(sg.convective_velocity(hfx, zi, t))
        if hfx <= 0:
            check(f"lead +{lead}h (HFX {hfx}) gives EXACTLY zero thermal",
                  w == 0.0, f"w* {w:.3f}")

    # Scaling: more heat and a deeper layer must both mean stronger thermals.
    weak = float(sg.convective_velocity(60, 1200, 20))
    strong = float(sg.convective_velocity(300, 3000, 32))
    boomer = float(sg.convective_velocity(450, 4000, 38))
    check("thermal strength scales with heating and depth",
          weak < strong < boomer, f"{weak:.2f} < {strong:.2f} < {boomer:.2f} m/s")
    check("and stays in the range pilots actually fly",
          0.8 < weak and boomer < 6.0,
          f"weak day {weak:.2f}, desert boomer {boomer:.2f} m/s")


def falsify_thermals():
    print("\n== FALSIFICATION: a stable night must never look flyable ==")
    # Every combination of night-time conditions: negative flux, shallow layer.
    for hfx in (-5, -18, -46, -100):
        for zi in (150, 270, 500):
            w = float(sg.convective_velocity(hfx, zi, 10.0))
            if w != 0.0:
                check(f"FALSIFY: HFX {hfx} zi {zi} produced lift", False,
                      f"w* {w:.3f} — a stable night must give zero")
                return
    check("no combination of negative heat flux produces thermals", True,
          "12 night-time cases, all exactly 0")

    # A deep layer with no heating is the trap: zi alone looks like a good day.
    w = float(sg.convective_velocity(-1, 4000, 35))
    check("FALSIFY: a deep layer with no heating is still zero", w == 0.0,
          f"w* {w:.3f} at zi=4000 — depth alone is not a thermal")


def chop():
    print("\n== B/S: organised thermals vs broken chop ==")
    # Dr Jack's threshold: B/S <= 5 means too broken to use.
    def bs(hfx, zi, t, u10):
        w = sg.convective_velocity(hfx, zi, t)
        return float(sg.buoyancy_shear_ratio(w, sg.friction_velocity(u10)))

    light = bs(300, 3000, 32, 4.0)
    windy = bs(400, 2500, 35, 18.0)
    check("strong thermals in light wind are organised", light > 12,
          f"B/S {light:.1f}")
    check("the SAME heating under 18 m/s is broken", windy < 5,
          f"B/S {windy:.1f} (measured 3.0, threshold 5)")
    check("...and that is a wind effect, not a heating one",
          bs(400, 2500, 35, 4.0) > 10 * windy,
          f"{bs(400, 2500, 35, 4.0):.0f} at 4 m/s vs {windy:.1f} at 18 m/s")

    # Monotone in wind: the ratio must fall as it blows harder.
    seq = [bs(250, 2500, 30, u) for u in (2, 5, 10, 15, 20)]
    check("B/S falls monotonically with wind speed",
          all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)),
          " > ".join(f"{v:.0f}" for v in seq))

    # No thermals means no organised thermals, whatever the wind does.
    check("no heating gives B/S zero, not a large number",
          bs(-20, 2000, 15, 3.0) == 0.0)


def ceiling():
    print("\n== top of lift: the boundary layer, capped by cloud base ==")
    # Blue day: no cloud reported, so the ceiling IS the boundary-layer top.
    cb = sg.cloud_base_agl(85000.0, np.nan, 30.0)
    check("no cloud reported gives no cloud base", not np.isfinite(cb))
    check("...so the ceiling is the boundary-layer top",
          float(sg.top_of_lift(2800.0, cb)) == 2800.0)

    # Cumulus below the PBL top: the climb stops at cloud base, not above it.
    cb2 = float(sg.cloud_base_agl(85000.0, 76000.0, 30.0))
    check("cloud base computes to a plausible height", 800 < cb2 < 1600,
          f"{cb2:.0f} m AGL")
    check("a cumulus base BELOW the PBL top caps the climb",
          float(sg.top_of_lift(2800.0, cb2)) == cb2,
          f"capped at {cb2:.0f} m rather than 2800 m")
    # And a base above the boundary layer must not RAISE the ceiling.
    check("a base above the PBL top does not raise the ceiling",
          float(sg.top_of_lift(1500.0, 3000.0)) == 1500.0)


def gusts():
    print("\n== gust factor: the launch-safety number ==")
    for lead, _zi, _hfx, _t, g, u in LIVE:
        gf = float(sg.gust_factor(g, u))
        check(f"lead +{lead}h gust factor is plausible", 1.0 <= gf < 3.0,
              f"{gf:.2f}x ({g:.1f} gusting over {u:.1f} mean)")
    # The live afternoon was smooth; the decoupled night was not.
    day = float(sg.gust_factor(7.3, 6.0))
    night = float(sg.gust_factor(9.8, 5.1))
    check("the night is gustier than the working afternoon", night > day * 1.3,
          f"{day:.2f}x by day vs {night:.2f}x at night")
    check("gust factor never goes below 1", float(sg.gust_factor(2.0, 8.0)) >= 1.0,
          "a gust cannot be slower than the mean")


def whole_field():
    print("\n== derive() on a grid, the shape the pipeline uses ==")
    n = 32
    rng = np.random.default_rng(0)
    fields = {
        "heat_flux": rng.uniform(-60, 350, (n, n)),
        "pbl_height": rng.uniform(120, 3200, (n, n)),
        "temp_c": rng.uniform(-5, 38, (n, n)),
        "wind_10m": rng.uniform(0.5, 22, (n, n)),
        "gust": rng.uniform(1, 30, (n, n)),
        "pressure_surface": np.full((n, n), 85000.0),
        "pressure_cloud_base": rng.uniform(60000, 84000, (n, n)),
        "cloud_cover": rng.uniform(0, 100, (n, n)),
    }
    out = sg.derive(fields)
    for k, v in out.items():
        finite = np.isfinite(v)
        # cloud_base is legitimately NaN where there is no cloud; everything
        # else must be a number everywhere.
        if k == "cloud_base":
            continue
        check(f"{k} is finite across the grid", finite.all(),
              f"{(~finite).sum()} non-finite of {v.size}")
    check("thermals are zero wherever the surface is not heating",
          (out["wstar"][fields["heat_flux"] <= 0] == 0).all())
    check("the ceiling never exceeds the boundary-layer top",
          (out["top_of_lift"] <= fields["pbl_height"] + 1e-9).all())


def decode_contract():
    print("\n== the pipeline -> browser contract survives quantisation ==")
    # Everything above tests the physics in float64. The browser reads BYTES,
    # and a quantisation bug there is invisible in both this file and the
    # panel — it just makes the numbers quietly wrong. So round-trip the
    # live-measured Bozeman values through the real writer and decode them
    # exactly as site/js/wind/soaring.js does.
    import tile_writer
    from config import SOAR_SCALES
    from PIL import Image
    import tempfile

    ny, nx = 256, 512          # enough to exercise the block-mean downsample
    step = SOAR_SCALES["wstar"] / 255
    with tempfile.TemporaryDirectory() as tmp:
        for zi, hfx, t, g, u in LIVE_ROWS:
            f = {
                "heat_flux": np.full((ny, nx), float(hfx)),
                "pbl_height": np.full((ny, nx), float(zi)),
                "temp_c": np.full((ny, nx), t),
                "wind_10m": np.full((ny, nx), u),
                "gust": np.full((ny, nx), g),
                "pressure_surface": np.full((ny, nx), 85000.0),
                "pressure_cloud_base": np.full((ny, nx), np.nan),
                "cloud_cover": np.zeros((ny, nx)),
            }
            d = sg.derive(f)
            tile_writer.write_soaring(tmp, "rt", d, SOAR_SCALES)
            a = np.array(Image.open(f"{tmp}/rt_soar.png")).astype(float)
            b = np.array(Image.open(f"{tmp}/rt_soar2.png")).astype(float)
            j, i = a.shape[0] // 2, a.shape[1] // 2
            w_dec = a[j, i, 0] / 255 * SOAR_SCALES["wstar"]
            gf_dec = b[j, i, 0] / 255 * SOAR_SCALES["gust"]
            w_true = float(d["wstar"][0, 0])
            gf_true = float(d["gust_factor"][0, 0])

            check(f"w* survives the round trip at HFX {hfx:+d}",
                  abs(w_dec - w_true) <= step,
                  f"{w_true:.3f} -> {w_dec:.3f} (step {step:.3f})")
            # THE one that matters: a stable night must decode to exactly zero.
            # Quantisation must not be able to invent a thermal.
            if hfx <= 0:
                check(f"...and zero stays EXACTLY zero at HFX {hfx:+d}",
                      w_dec == 0.0, f"{w_dec:.4f}")
            check(f"gust factor survives at HFX {hfx:+d}",
                  abs(gf_dec - gf_true) < 0.03,
                  f"{gf_true:.2f} -> {gf_dec:.2f}")

        # Blue sky must not decode as cloud on the deck. Cloud base uses 0 as a
        # presence flag precisely because 0 is a real height.
        f["pressure_cloud_base"] = np.full((ny, nx), np.nan)
        tile_writer.write_soaring(tmp, "blue", sg.derive(f), SOAR_SCALES)
        a = np.array(Image.open(f"{tmp}/blue_soar.png"))
        check("no cloud reported encodes as the blue-sky flag, not zero height",
              (a[:, :, 2] == 0).all(), "all cells flagged blue")


def main():
    thermal_strength()
    falsify_thermals()
    chop()
    ceiling()
    gusts()
    whole_field()
    decode_contract()
    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
