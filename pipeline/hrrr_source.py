"""Open the dynamical.org HRRR virtual icechunk store and pick a usable init.

Two things here are not obvious and were both paid for with failed reads:

1. ``dynamical_catalog.open()`` loads *every* collection in the STAC catalog and
   raises if any one of them 404s. An unrelated ECMWF collection was broken
   upstream on 2026-08-10, which took the whole pipeline down. We resolve the
   single collection we need instead.

2. This is a *virtual* store: its chunks are byte-range references into the
   public ``noaa-hrrr-bdp-pds`` bucket. icechunk refuses to follow them unless
   the container is authorized at open time.
"""

from __future__ import annotations

import json
import logging
import urllib.request

import icechunk
import numpy as np
import xarray as xr

from config import DATASET_ID, PROBE_WINDOW

log = logging.getLogger(__name__)

STAC_COLLECTION = "https://stac.dynamical.org/{ds}/collection.json"


def collection_config(dataset_id: str = DATASET_ID) -> dict:
    """Resolve one STAC collection into icechunk storage parameters."""
    url = STAC_COLLECTION.format(ds=dataset_id)
    with urllib.request.urlopen(url, timeout=60) as resp:
        collection = json.load(resp)

    asset = collection["assets"]["icechunk"]
    bucket, prefix = asset["href"][len("s3://") :].rstrip("/").split("/", 1)
    containers = asset.get("icechunk:virtual_chunk_containers") or []
    return {
        "bucket": bucket,
        "prefix": prefix,
        "region": asset.get("icechunk:storage", {}).get("region", "us-west-2"),
        "virtual_prefixes": [
            c["url_prefix"] if isinstance(c, dict) else c for c in containers
        ],
    }


def open_groups(dataset_id: str = DATASET_ID):
    """Return (surface_ds, model_level_ds) as lazy xarray Datasets."""
    cfg = collection_config(dataset_id)
    storage = icechunk.s3_storage(
        bucket=cfg["bucket"],
        prefix=cfg["prefix"],
        region=cfg["region"],
        anonymous=True,
    )
    authorize = (
        icechunk.containers_credentials(
            {p: icechunk.s3_anonymous_credentials() for p in cfg["virtual_prefixes"]}
        )
        if cfg["virtual_prefixes"]
        else None
    )
    repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=authorize)
    store = repo.readonly_session("main").store
    sfc = xr.open_zarr(store, chunks=None, consolidated=False)
    ml = xr.open_zarr(store, chunks=None, group="model_level", consolidated=False)
    return sfc, ml


def pick_init(ml, sfc, leads, requested: str | None = None, max_steps_back: int = 4):
    """Choose an init_time whose every lead the build reads is in, in both groups.

    The dataset lists an init as soon as ingest starts, so the newest one is
    routinely all-NaN — 2026-08-10T17:00Z was empty while 16:00Z was complete.
    Probe, and step back an hour at a time.

    Probing only the model-level wind at the dataset's final lead was not
    enough. Twice (2026-11-18 18Z, 2026-11-23 00Z) an init passed that probe,
    the build read all 18 leads, and then failed at the very end with
    "stability field is entirely NaN". Boundary-layer height is a surface-group
    field, and the final lead is not one the build reads, so the probe now
    covers every lead the build will read, in both groups.
    """
    if requested and requested != "auto":
        init = np.datetime64(requested)
        _probe(ml, sfc, init, leads)
        return init

    inits = ml.init_time.values
    for i in range(1, max_steps_back + 2):
        init = inits[-i]
        try:
            _probe(ml, sfc, init, leads)
            log.info("using init %s", init)
            return init
        except Exception as exc:  # noqa: BLE001 - any read failure means incomplete
            log.warning("init %s unusable (%s); stepping back", init, exc)
    raise RuntimeError(f"no complete init in the last {max_steps_back + 1} cycles")


def _probe(ml, sfc, init, leads):
    """Raise unless a small window is readable at every lead, in both groups."""
    ys, xs = PROBE_WINDOW
    window = {"lead_time": slice(0, leads), "y": slice(*ys), "x": slice(*xs)}
    fields = {
        "model-level wind": ml.wind_u.sel(init_time=init).isel(model_level=0, **window),
        "boundary-layer height": sfc.planetary_boundary_layer_height_surface
        .sel(init_time=init).isel(**window),
    }
    for what, field in fields.items():
        slab = field.transpose("lead_time", ...).values
        empty = np.isnan(slab.reshape(slab.shape[0], -1)).all(axis=1)
        if empty.any():
            raise ValueError(f"{what} is all-NaN at lead index "
                             f"{np.flatnonzero(empty).tolist()} (ingest incomplete)")
