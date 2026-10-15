"""
Fold assignment along the valley axis.

Cuts are absolute kilometres, identical for every scene, so a given stretch of
river is either training or validation in all of them. The two ends are always
training because one scene is missing there.

Stations are computed from the outlines in NOVEMBER coordinates - that is the
system the valley axis lives in. The per-scene files are row-aligned with the
november ones, so the fold label transfers by position.
"""
import numpy as np
import geopandas as gpd
from shapely.ops import linemerge
from shapely import line_locate_point

import config


def axis(path=None):
    g = gpd.read_file(path or config.VALLEY_AXIS).to_crs(config.CRS)
    return linemerge(g.geometry.union_all()) if len(g) > 1 else g.geometry.iloc[0]


def station_km(geoms, line=None):
    """distance along the valley axis in km, for geometries in november coords"""
    line = line if line is not None else axis()
    cent = gpd.GeoSeries(geoms, crs=config.CRS).centroid.values
    return line_locate_point(line, cent) / 1000.0


def label(station):
    """fold label for one or many stations"""
    s = np.atleast_1d(np.asarray(station, float))
    out = np.full(s.shape, "", dtype=object)

    for a, b in config.ALWAYS_TRAIN:
        out[(s >= a) & (s < b)] = "always_train"

    out[(s >= config.TEST_FROM) & (s < config.TEST_TO)] = "test"

    edges = config.FOLD_EDGES
    for i in range(len(edges) - 1):
        m = (s >= edges[i]) & (s < edges[i + 1]) & (out == "")
        out[m] = f"fold{i + 1}"

    out[out == ""] = "always_train"       # anything beyond the defined ranges
    return out if out.size > 1 else out[0]


def training_folds(validation_fold):
    """which labels count as training when validating on `validation_fold`"""
    all_folds = [f"fold{i + 1}" for i in range(len(config.FOLD_EDGES) - 1)]
    return ["always_train"] + [f for f in all_folds if f != validation_fold]


def summary(gdf_nov):
    """count outlines per fold, for a gdf in november coordinates"""
    import collections
    lab = label(station_km(gdf_nov.geometry))
    return dict(sorted(collections.Counter(lab).items()))
