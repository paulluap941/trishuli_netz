"""
Splits the data into folds along the river.

The cuts are absolute kilometres, the same in every scene. I first wanted to
take the last 20 % of each scene, but that hits different ground: march runs
0.2-24.7 km, november 0-26.2, february 7.7-26.2, so "last 20 %" starts at
19.8, 20.9 and 22.5 km. Same stretch would be validation in one scene and
training in another.

Stations come from the outlines in november coordinates, because that is the
system the valley axis is in. The per-scene files have the same row order, so
the label carries over by position.
"""
import collections
import numpy as np
import geopandas as gpd
from shapely.ops import linemerge
from shapely import line_locate_point

import config


def axis(path=None):
    g = gpd.read_file(path or config.VALLEY_AXIS).to_crs(config.CRS)
    return linemerge(g.geometry.union_all()) if len(g) > 1 else g.geometry.iloc[0]


def station_km(geoms, line=None):
    line = line if line is not None else axis()
    cent = gpd.GeoSeries(geoms, crs=config.CRS).centroid.values
    return line_locate_point(line, cent) / 1000.0


def label(station):
    s = np.atleast_1d(np.asarray(station, float))
    out = np.full(s.shape, "", dtype=object)

    for a, b in config.ALWAYS_TRAIN:
        out[(s >= a) & (s < b)] = "always_train"

    out[(s >= config.TEST_FROM) & (s < config.TEST_TO)] = "test"

    edges = config.FOLD_EDGES
    for i in range(len(edges) - 1):
        m = (s >= edges[i]) & (s < edges[i + 1]) & (out == "")
        out[m] = f"fold{i + 1}"

    out[out == ""] = "always_train"     # anything outside the defined ranges
    return out if out.size > 1 else out[0]


def training_folds(validation_fold):
    all_folds = [f"fold{i + 1}" for i in range(len(config.FOLD_EDGES) - 1)]
    return ["always_train"] + [f for f in all_folds if f != validation_fold]


def summary(gdf_nov):
    lab = label(station_km(gdf_nov.geometry))
    return dict(sorted(collections.Counter(lab).items()))
