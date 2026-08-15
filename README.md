# hrrr-wind-atlas

Turns NOAA's HRRR forecast into a compact PNG atlas of 3D wind, rebuilt hourly
and published to object storage. It is the data half of a 3D wind viewer; the
renderer lives elsewhere.

The output is a set of quantised PNG tiles plus a `meta.json` describing the
grid, the height ladder and the per-level scales. A browser can sample wind at
any point and height by reading a texture, with no server in the loop.

## What it does

1. Reads HRRR **model levels** from the [dynamical.org](https://dynamical.org)
   icechunk store, anonymously — no credentials, no account.
2. Interpolates each column onto a fixed **height-above-ground ladder** using
   that column's own `geopotential_height − geopotential_height_surface`. Model
   levels are terrain-following, which is what makes wind near the ground mean
   anything over mountains.
3. Regrids from HRRR's Lambert conformal projection to Web Mercator, rotating
   the winds from grid-relative to earth-relative.
4. Derives soaring parameters (Deardorff w\*, buoyancy/shear ratio, top of lift,
   gust factor) using RASP/BLIPMAP vocabulary, so the numbers mean what a pilot
   already expects them to mean.
5. Quantises everything to 8-bit PNG tiles with a one-cell gutter, so hardware
   bilinear filtering never samples across a tile edge.

Roughly 650 MB per run for 18 forecast hours, built in about 20 minutes.

## Why this repo is public

GitHub Actions minutes are unlimited and free on public repositories. An hourly
20-minute build is ~14,600 minutes a month, which no private free tier covers.
The pipeline reads public NOAA data anonymously and contains no credentials —
the only secrets are the upload keys, which are repository secrets.

**The cron only fires from the default branch.** GitHub reads `schedule:` from
the workflow file on the default branch only; a cron on any other branch is
silently dead and produces no runs at all, not even failing ones. If the atlas
stops updating, check `event=schedule` run counts before anything else.

## Running it

```sh
pip install -r requirements.txt
PYTHONPATH=pipeline python pipeline/build.py --out out --leads 18 --workers 11

# Tests are plain scripts, not pytest: they print PASS/FAIL and exit non-zero.
PYTHONPATH=pipeline python test/test_soaring.py
# This one needs a slab of real model-level data, so it is run by hand rather
# than in CI:
PYTHONPATH=pipeline python test/test_agl_interp.py <slab.npz>
```

`--leads` is the number of forecast hours; `--workers` the read concurrency
(reads are network-latency-bound, so more workers help well past the core
count). `--init` defaults to `auto`, which probes the newest initialisation and
steps back if it is not yet populated — the newest listed init is routinely all
NaN for a while after it appears.

## Publishing

The workflow syncs `out/` to an S3-compatible bucket with `aws s3 sync
--delete`. That is genuinely incremental: terrain tiles are identical between
runs and get skipped, so only the wind actually uploads. `meta.json` goes last
and with a short cache lifetime, because it names which leads exist — a stale
copy points a reader at tiles that have already been deleted.

Configure via repository secrets `CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, and variables `R2_BUCKET`, `WIND_DATA_URL` (public
base URL, for the post-publish check) and `WIND_SITE_URL` (the reader's origin,
so the CORS check asks the question a browser would).

## Tests

`test/test_agl_interp.py` runs against real data: the interpolated rungs
reproduce their targets to 0.0000 m, winds stay inside their bracketing levels,
and the surface rung substitution behaves. `test/test_soaring.py` pins the
soaring physics, including that `w*` is exactly zero when the surface heat flux
is not positive — that is physics, not a guard.

Two things worth knowing if you change the interpolation:

- **Interpolate components, never speed or direction.** A column whose u and v
  reverse through a shear layer legitimately passes through near-calm; averaging
  speed hides that and averaging direction is meaningless across the wrap.
- **The 10 m rung is extrapolated in essentially every column**, because model
  level 1 sits above 10 m everywhere in the domain. It uses HRRR's own 10 m
  diagnostic instead. Check whether a rung is extrapolated per column, not
  domain-wide: a domain-wide minimum hides that it is above the target almost
  everywhere.
