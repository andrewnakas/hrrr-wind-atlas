"""Quantise the AGL wind field and write it as PNG tiles.

Layout, and why:

* One PNG per lead per tile. RGB carries u, v and w; there is no alpha, because
  a validity mask would be identical in every frame — it ships once as a
  separate image instead of 19 copies.

* A tile is TILE cells wide plus a one-cell GUTTER copied from its neighbours.
  Hardware bilinear sampling reads one texel past the cell it lands in, so
  without the gutter the frontend would blend a tile's edge against whatever
  happens to be adjacent in the atlas, and the seams between tiles would show
  as lines of wrong wind.

* The AGL levels of one tile are stacked vertically into a single image, so a
  tile is one HTTP request and one texture upload regardless of how many
  heights it carries.
"""

from __future__ import annotations

import io
import json
import os
import warnings

import numpy as np
from PIL import Image

from config import (
    AGL_LADDER,
    GUTTER,
    OUT_NX,
    OUT_NY,
    OVERVIEW_FACTOR,
    SOAR_FACTOR,
    STAB_FACTOR,
    SCALE_PAD,
    TILE,
    W_SCALE_MIN,
)


def lead_percentiles(ug, vg, wg):
    """Per-level 99.99th percentiles of |value| for ONE lead.

    The scales have to be global — a per-lead scale makes the wind's brightness
    pulse as the animation advances — but computing them does not require every
    lead in memory at once. Reducing each lead to 3 x nk numbers as it is read,
    and taking the max across leads, gives the same answer for 17 floats per
    lead instead of 0.53 GB.

    (Not exactly the same: the max of per-lead p99.99 is not the p99.99 of the
    pooled set. It is very slightly larger, which means very slightly more
    quantisation headroom — the safe direction, and far below one byte step.)
    """
    out = {}
    for key, field in (("u", ug), ("v", vg), ("w", wg)):
        out[key] = [float(np.nanpercentile(np.abs(field[:, :, k]), 99.99))
                    for k in range(field.shape[2])]
    return out


def scales_from_percentiles(per_lead):
    """Combine per-lead percentiles into the shipped scales."""
    keys = ("u", "v", "w")
    nk = len(per_lead[0]["u"])
    out = {}
    for key in keys:
        floor = W_SCALE_MIN if key == "w" else 0.0
        out[key] = [
            float(max(np.ceil(max(p[key][k] for p in per_lead) * SCALE_PAD), floor))
            for k in range(nk)
        ]
    return out


def compute_scales(u, v, w):
    """Symmetric per-level scales, from the 99.99th percentile of |value|.

    Per level rather than per tile: a tile-local scale would make the same wind
    quantise differently either side of a tile edge, which is visible as a seam
    once particles cross it.
    """
    def scale_of(field, floor=0.0):
        out = []
        for k in range(field.shape[2]):
            m = np.nanpercentile(np.abs(field[:, :, k]), 99.99)
            out.append(float(max(np.ceil(m * SCALE_PAD), floor)))
        return out

    return {
        "u": scale_of(u),
        "v": scale_of(v),
        "w": scale_of(w, W_SCALE_MIN),
    }


def quantise(field, scales):
    """Map each level to 0..255 around its own symmetric scale."""
    out = np.empty(field.shape, dtype=np.uint8)
    for k in range(field.shape[2]):
        s = scales[k]
        norm = np.clip(field[:, :, k] / s * 0.5 + 0.5, 0.0, 1.0)
        # Invalid cells (outside the HRRR domain) encode as zero wind rather
        # than as noise, and the mask tells the frontend to ignore them.
        norm = np.where(np.isfinite(norm), norm, 0.5)
        out[:, :, k] = (norm * 255).astype(np.uint8)
    return out


def _tile_slice(arr, x0, y0, w, h):
    """Extract a tile with its gutter, clamping at the domain edge."""
    xs = slice(max(0, x0 - GUTTER), min(OUT_NX, x0 + w + GUTTER))
    ys = slice(max(0, y0 - GUTTER), min(OUT_NY, y0 + h + GUTTER))
    return arr[ys, xs], xs, ys


def write_lead_tiles(out_dir, lead_name, qu, qv, qw):
    """Write every tile for one lead. Returns total bytes written."""
    os.makedirs(out_dir, exist_ok=True)
    nk = qu.shape[2]
    total = 0
    for ty in range(OUT_NY // TILE):
        for tx in range(OUT_NX // TILE):
            x0, y0 = tx * TILE, ty * TILE
            su, xs, ys = _tile_slice(qu, x0, y0, TILE, TILE)
            sv, _, _ = _tile_slice(qv, x0, y0, TILE, TILE)
            sw, _, _ = _tile_slice(qw, x0, y0, TILE, TILE)
            h = ys.stop - ys.start
            wdt = xs.stop - xs.start
            strip = np.zeros((h * nk, wdt, 3), dtype=np.uint8)
            for k in range(nk):
                strip[k * h:(k + 1) * h, :, 0] = su[:, :, k]
                strip[k * h:(k + 1) * h, :, 1] = sv[:, :, k]
                strip[k * h:(k + 1) * h, :, 2] = sw[:, :, k]
            path = os.path.join(out_dir, f"{lead_name}_{tx}_{ty}.png")
            Image.fromarray(strip, "RGB").save(path, optimize=True)
            total += os.path.getsize(path)
    return total


def write_overview(out_dir, lead_name, qu, qv, qw):
    """Whole-domain image at reduced resolution, always resident in the client.

    This is what a point reads before its native tile has loaded, so the wind
    is never simply absent; it just sharpens.
    """
    f = OVERVIEW_FACTOR
    su, sv, sw = qu[::f, ::f], qv[::f, ::f], qw[::f, ::f]
    h, wdt, nk = su.shape
    strip = np.zeros((h * nk, wdt, 3), dtype=np.uint8)
    for k in range(nk):
        strip[k * h:(k + 1) * h, :, 0] = su[:, :, k]
        strip[k * h:(k + 1) * h, :, 1] = sv[:, :, k]
        strip[k * h:(k + 1) * h, :, 2] = sw[:, :, k]
    path = os.path.join(out_dir, f"{lead_name}_overview.png")
    Image.fromarray(strip, "RGB").save(path, optimize=True)
    return os.path.getsize(path)


def write_stability(out_dir, lead_name, pblh, n_brunt, scales):
    """Boundary-layer depth and low-level stability, for the solver's alpha knob.

    One tiny image per lead — a 64x40 field of ~80 km cells, a few KB.

    R = PBL height, G = Brunt-Vaisala frequency N. B is unused and zero (PNG has
    no two-channel mode worth the special case here).

    N is scaled rather than shipped raw because its useful range is tiny: 0.005
    for a convective afternoon against 0.035 for a strong nocturnal inversion,
    so a byte across 0..N_SCALE resolves it to about 0.0002 s^-1.
    """
    # Far coarser than the wind overview. The browser applies ONE alpha and ONE
    # lid to a ~50 km solve window, and both fields vary on synoptic scales, so
    # anything finer is averaged away before it is used — it would only cost
    # bytes. Block-MEAN rather than strided sampling, so a coarse cell is
    # representative of its area instead of of one arbitrary point in it.
    f = STAB_FACTOR
    ny, nx = pblh.shape
    ch, cw = ny // f, nx // f
    trim_p = pblh[:ch * f, :cw * f].reshape(ch, f, cw, f)
    trim_n = n_brunt[:ch * f, :cw * f].reshape(ch, f, cw, f)
    # A block that is ENTIRELY outside the domain averages to NaN, so the mean
    # has to be taken before it can be used as the fill — hence two steps rather
    # than one. (`Mean of empty slice` is exactly that case, and suppressing the
    # warning without fixing the fill would have left 58 dead cells.)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sp = np.nanmean(trim_p, axis=(1, 3))
            sn = np.nanmean(trim_n, axis=(1, 3))

    # Out-of-domain cells come back NaN from the regrid. Casting NaN to uint8 is
    # undefined and lands on 0, which would read back as "zero PBL height and
    # perfectly neutral air" — a plausible-looking value that is not data, and
    # one the browser cannot tell from a real reading. Fill from the domain mean
    # so an edge cell degrades to typical rather than to a lie.
    for arr in (sp, sn):
        good = np.isfinite(arr)
        if not good.any():
            raise ValueError("stability field is entirely NaN")
        np.copyto(arr, arr[good].mean(), where=~good)

    img = np.zeros((ch, cw, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip(sp / scales["pblh"] * 255.0, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(sn / scales["n"] * 255.0, 0, 255).astype(np.uint8)
    path = os.path.join(out_dir, f"{lead_name}_stab.png")
    Image.fromarray(img, "RGB").save(path, optimize=True)
    return os.path.getsize(path)


def write_soaring(out_dir, lead_name, soar, scales):
    """The soaring forecast for one lead — two small RGB images.

    Read AT A POINT when a pilot clicks their launch, so unlike the stability
    field (which the solver averages to one number per window) the detail here
    is the product. 5 km cells, about 1.3 MB per lead.

    Split across two images because there are six quantities and PNG carries
    three channels; a two-channel mode would save nothing worth the special
    case.

        _soar   R = w* (thermal strength)  G = top of lift  B = cloud base
        _soar2  R = gust factor            G = B/S ratio    B = cloud cover

    NaN handling matters more here than for wind. Out-of-domain cells regrid to
    NaN, and casting NaN to uint8 lands on 0 — which for w* reads back as "no
    thermals" and for cloud base as "cloud on the deck". Both are plausible
    forecasts rather than obvious corruption, which is exactly what makes them
    dangerous. Cloud base has no sentinel available (0 is a real value), so it
    is stored as a PRESENCE flag: 0 means blue sky, and any other value is a
    real height.
    """
    f = SOAR_FACTOR
    ny, nx = soar["wstar"].shape
    ch, cw = ny // f, nx // f

    def block_mean(a):
        t = np.asarray(a, dtype=np.float64)[:ch * f, :cw * f]
        t = t.reshape(ch, f, cw, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(t, axis=(1, 3))

    def quant(a, ceiling, fill=None):
        v = block_mean(a)
        good = np.isfinite(v)
        if not good.any():
            v = np.zeros_like(v)
        else:
            # Fill from the domain mean, so an edge cell degrades to typical
            # rather than to a number that reads as a real forecast.
            np.copyto(v, v[good].mean() if fill is None else fill, where=~good)
        return np.clip(v / ceiling * 255.0, 0, 255).astype(np.uint8)

    # Cloud base: absent means blue sky, which is information, not a gap.
    base = block_mean(soar["cloud_base"])
    base_q = np.where(np.isfinite(base),
                      np.clip(base / scales["base"] * 255.0, 1, 255), 0).astype(np.uint8)

    a = np.zeros((ch, cw, 3), dtype=np.uint8)
    a[:, :, 0] = quant(soar["wstar"], scales["wstar"], fill=0.0)
    a[:, :, 1] = quant(soar["top_of_lift"], scales["top"])
    a[:, :, 2] = base_q

    b = np.zeros((ch, cw, 3), dtype=np.uint8)
    b[:, :, 0] = quant(soar["gust_factor"], scales["gust"], fill=1.0)
    b[:, :, 1] = quant(soar["bs_ratio"], scales["bs"], fill=0.0)
    b[:, :, 2] = quant(soar["cloud_cover"], scales["cover"], fill=0.0)

    total = 0
    for name, img in ((f"{lead_name}_soar.png", a), (f"{lead_name}_soar2.png", b)):
        path = os.path.join(out_dir, name)
        Image.fromarray(img, "RGB").save(path, optimize=True)
        total += os.path.getsize(path)
    return total


def write_mask(out_dir, valid):
    """Validity mask. Time-invariant, so it ships once."""
    img = (valid.astype(np.uint8) * 255)
    path = os.path.join(out_dir, "mask.png")
    Image.fromarray(img, "L").save(path, optimize=True)
    return os.path.getsize(path)


def write_meta(out_dir, meta):
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return os.path.getsize(os.path.join(out_dir, "meta.json"))
