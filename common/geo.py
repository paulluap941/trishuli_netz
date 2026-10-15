"""Small things that several notebooks and modules need, so they only exist once."""
import numpy as np
import geopandas as gpd

import config


def diameter(gdf):
    """equivalent circle diameter in m, the size I report everywhere"""
    return 2 * np.sqrt(gdf.area / np.pi)


def load_outlines(scene, nov=False, min_d=None):
    """
    Mapped boulders for one scene.

    nov=False gives the scene's own coordinates, which match its image.
    nov=True gives the same rows moved into november's system, which is where
    the valley axis is. Row order is identical in both, the fold label carries
    over by position.
    """
    g = gpd.read_file(config.outlines(scene, nov=nov)).to_crs(config.CRS)
    g["d_m"] = diameter(g)
    if min_d is not None:
        g = g[g.d_m >= min_d].reset_index(drop=True)
    return g
