"""Pipeline constants — the contract between the build and the frontend.

Every number here that isn't arbitrary was measured; see the comments.
"""

DATASET_ID = "noaa-hrrr-forecast-18-hour-virtual"

# Fixed heights above ground, in metres, that particles ride. Because HRRR's
# native model levels are terrain-following (level 1 sits at 9.9-11.5 m AGL
# across the whole CONUS domain, from -81 m to 3995 m surface elevation), we can
# interpolate each column onto this ladder exactly. The frontend then gets a
# regular (x, y, rung) field with AGL >= 0 everywhere, which is what makes
# "particle z = terrain + AGL" unable to clip.
# Spacing is roughly logarithmic: fine near the surface where the shear is
# steep, coarse aloft where the flow is smooth. The upper rungs reach the
# mid-troposphere so the whole depth of the weather is visible, not just the
# boundary layer.
#
# HRRR's PRESSURE levels are deliberately not used for this. They are not
# terrain-following: 1000, 950, 900, 850 and 800 hPa all sit below Rocky
# Mountain terrain (~110 m to ~1985 m MSL against 2000-4000 m ground), so
# particles riding them would draw inside mountains — exactly the v1 bug this
# rebuild exists to make impossible. The model levels are terrain-following and
# span 11 m to 25,836 m AGL, which covers the same air without that failure.
AGL_LADDER = [
    10.0, 25.0, 50.0, 100.0, 200.0, 350.0, 550.0, 800.0, 1200.0, 2000.0,
    3000.0, 4200.0, 5800.0, 7600.0, 9500.0, 11500.0, 13500.0,
]

# Model levels to read. Measured against real columns: reaching 13.5 km AGL
# needs 38 levels over the Rockies (surface 2095 m), 35 over the Plains and 34
# at the Gulf coast. 40 leaves margin without reading the whole stratosphere.
N_MODEL_LEVELS = 40

# Source grid (HRRR Lambert Conformal Conic) — from the dataset's own coords.
HRRR_NX, HRRR_NY = 1799, 1059
LCC_PROJ = (
    "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 "
    "+a=6371229 +b=6371229 +units=m +no_defs"
)
LCC_X0, LCC_X1 = -2697520.142521929, 2696479.857478071
LCC_Y0, LCC_Y1 = -1587306.152556665, 1586693.847443335
# Grid-relative to earth-relative wind rotation (single-meridian LCC).
ROTCON = 0.6225146
LCC_LON0 = -97.5

# Output grid: mercator-spaced lon/lat covering CONUS. 2048x1280 gives ~2.5-2.9 km
# ground cells, a slight oversample of HRRR's native 3 km (never undersamples).
OUT_NX, OUT_NY = 2048, 1280
WEST, EAST = -125.5, -66.5
SOUTH, NORTH = 23.5, 50.5

# Tiling. 256-cell tiles => 8x5 = 40 tiles per lead. The 1-cell gutter is baked
# from neighbouring tiles so hardware bilinear never samples across a seam.
TILE = 256
GUTTER = 1
OVERVIEW_FACTOR = 4

# Measured 2026-08-10: 19.3 MB of native tiles + 2.1 MB overview per lead,
# => ~406 MB for 19 leads, well inside the ~1 GB Pages artifact limit.
#
# 18 is what the site builds: it is the forecast span a pilot plans against
# (this afternoon's flight AND tomorrow morning), and it is what the dataset
# guarantees on every hourly cycle. Measured on CI: ~20 min of a 45-min timeout
# and ~646 MB of a 900 MB guard, so leads are the first thing to trim if either
# ever gets tight.
N_LEADS = 18

# Quantisation headroom over the 99.99th percentile of |value|.
SCALE_PAD = 1.05
W_SCALE_MIN = 1.0  # m/s; keep vertical motion resolvable in calm frames

# Physical constants for omega (Pa/s) -> w (m/s), and for potential temperature.
G = 9.80665
R_DRY = 287.06
CP = 1004.6

# Stability product, for the solver's alpha knob (flow over vs around terrain).
#
# The layer N is measured over. It matches the depth the browser's solve window
# actually covers, so the stability reported is the stability of the air the
# solver is correcting — not of some deeper column it never touches.
STAB_TOP_M = 2000.0
# Quantisation ceilings. PBL height reaches ~2300 m over CONUS in summer
# (measured 2026-08-12: max 2313 m, p99 1301 m); 4000 leaves headroom for deep
# convective afternoons without wasting much of the byte.
PBLH_SCALE = 4000.0
# N's useful range is small — 0.005 s^-1 for a convective afternoon against
# ~0.035 for a strong nocturnal inversion — so a tight ceiling is what makes a
# single byte resolve it to ~0.0002 s^-1.
N_BRUNT_SCALE = 0.05
# Downsample factor for the stability image, relative to the full output grid.
# 32 gives a 64x40 field (~80 km cells), which is finer than the synoptic scale
# these quantities actually vary on and far finer than the ~50 km solve window
# that consumes them. Anything finer is averaged away before it is used.
STAB_FACTOR = 32

# Soaring product, for the flight-planning panel.
#
# MUCH finer than the stability field, and for a reason: the browser reduces
# stability to ONE alpha and ONE lid per ~50 km solve window, so detail there is
# averaged away, while this product is read AT A POINT when a pilot clicks their
# launch. Terrain-scale variation is the whole value.
#
# Factor 2 gives 1024x640, about 5 km cells — HRRR's own resolution is 3 km, so
# going finer would only interpolate. Measured cost at 18 leads: ~24 MB against
# a 900 MB guard.
SOAR_FACTOR = 2

# Quantisation ceilings, from what these quantities actually reach.
SOAR_SCALES = {
    # Deardorff w*. 3.7 m/s is a desert boomer; 6 leaves headroom for anywhere.
    "wstar": 6.0,
    # Top of lift and cloud base, metres AGL. HRRR PBL height maxes near 2300 m
    # over CONUS in summer, but cloud base over the Rockies can run higher.
    "top": 6000.0,
    "base": 6000.0,
    # Gust over mean. Above ~3 it is academic — nobody is flying.
    "gust": 4.0,
    # B/S. The threshold that matters is 5, and everything above ~40 is
    # "thoroughly organised", so a tight ceiling spends the byte where the
    # decision actually changes.
    "bs": 50.0,
    "cover": 100.0,     # percent
    "wind": 40.0,       # m/s at 10 m
}

# Interior window used to probe whether an init has ingested.
PROBE_WINDOW = ((520, 540), (890, 910))
