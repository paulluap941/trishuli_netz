"""
Cuts the scenes into tiles and burns the outlines into label rasters.

Each scene stays in its own coordinate system. I only resample to 0.125 m/px,
I do not warp anything onto a common grid. Two reasons: the labels then sit
exactly where I drew them, and the data gets interpolated once instead of
twice. For training the scenes do not need to line up with each other anyway.

Three files per tile:
    <scene>_<col>_<row>.tif           image, uint16, raw values
    <scene>_<col>_<row>_masks.tif     label raster, one id per boulder
    <scene>_<col>_<row>_corridor.tif  uint8, 1 where I actually looked
"""
import json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import box
from shapely import prepare, intersects

import config
from common.offset import Offset
from common import folds, geo


def corridor_in_scene(scene):
    geom = gpd.read_file(config.corridor(scene)).to_crs(config.CRS).union_all()
    if scene == config.REFERENCE:
        return geom
    return Offset(config.VALLEY_AXIS, scene).nov_to_scene(geom)


def grid(bounds, res=None, tile=None, overlap=None):
    """tile origins, stepping by the inner width so the inner regions tile the area"""
    res = res or config.RES
    tile = tile or config.TILE
    overlap = overlap or config.OVERLAP

    step = (tile - overlap) * res
    size = tile * res
    pad = overlap * res / 2
    x0, y0, x1, y1 = bounds
    xs = np.arange(np.floor((x0 - pad) / step) * step, x1 + pad, step)
    ys = np.arange(np.floor((y0 - pad) / step) * step, y1 + pad, step)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            yield i, j, (x, y, x + size, y + size)


def transform(bounds, res):
    return from_origin(bounds[0], bounds[3], res, res)


def read_window(path, bounds, size):
    """read a window and resample it to size x size"""
    with rasterio.open(path) as r:
        win = from_bounds(*bounds, transform=r.transform)
        return r.read(1, window=win, out_shape=(size, size),
                      resampling=Resampling.bilinear,
                      boundless=True, fill_value=0)


def burn(geoms, bounds, size, res, dtype=np.uint16):
    """rasterise (geometry, value) pairs onto the tile grid"""
    return rasterize(geoms, out_shape=(size, size), transform=transform(bounds, res),
                     fill=0, dtype=dtype)


def write_raster(path, array, bounds, res):
    with rasterio.open(path, "w", driver="GTiff",
                       height=array.shape[0], width=array.shape[1], count=1,
                       dtype=array.dtype, crs=config.CRS,
                       transform=transform(bounds, res),
                       compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256) as w:
        w.write(array, 1)


def inside(gdf, geom):
    """rows whose centroid lies in geom, using the spatial index"""
    hit = gdf.iloc[list(gdf.sindex.query(geom, predicate="intersects"))]
    return hit[hit.geometry.centroid.within(geom)]


def build_scene(scene, out_dir=None, res=None, tile=None, overlap=None, min_d=None,
                verbose=True):
    out_dir = out_dir or config.ARCHIVE
    res = res or config.RES
    tile = tile or config.TILE
    overlap = overlap or config.OVERLAP
    min_d = min_d or config.MIN_TRAINING
    out_dir.mkdir(parents=True, exist_ok=True)

    img = config.image(scene)
    corr = corridor_in_scene(scene)
    prepare(corr)

    own = geo.load_outlines(scene, min_d=min_d)
    nov = geo.load_outlines(scene, nov=True, min_d=min_d)
    assert len(own) == len(nov), f"{scene}: the two outline files are out of sync"
    own["fold"] = folds.label(folds.station_km(nov.geometry))
    own.sindex

    half = overlap * res / 2
    area = (tile * res) ** 2
    index = []

    for col, row, b in grid(corr.bounds, res, tile, overlap):
        bx = box(*b)
        if not intersects(corr, bx):
            continue
        share = corr.intersection(bx).area / area
        if share < config.MIN_CORRIDOR:
            continue

        image = read_window(img, b, tile)
        if (image > 0).mean() < 0.5:
            continue                      # mostly outside the image footprint

        hit = own.iloc[list(own.sindex.query(bx, predicate="intersects"))]
        labels = burn(((g, i) for i, g in enumerate(hit.geometry.values, start=1)),
                      b, tile, res) if len(hit) else np.zeros((tile, tile), np.uint16)

        mask = burn([(corr.intersection(bx), 1)], b, tile, res, dtype=np.uint8)
        # a boulder sitting on the corridor edge would otherwise be half ignored
        mask |= (labels > 0).astype(np.uint8)

        name = f"{scene}_{col:03d}_{row:03d}"
        write_raster(out_dir / f"{name}.tif", image.astype(np.uint16), b, res)
        write_raster(out_dir / f"{name}_masks.tif", labels, b, res)
        write_raster(out_dir / f"{name}_corridor.tif", mask, b, res)

        # fold of the tile: whatever most of its boulders say. empty tiles fall
        # back to the station of their centre, good enough since there is
        # nothing in them that could leak.
        inner = box(b[0] + half, b[1] + half, b[2] - half, b[3] - half)
        sel = inside(own, inner)
        if len(sel):
            fold = sel["fold"].mode().iloc[0]
        else:
            centre = gpd.GeoSeries.from_xy([(b[0] + b[2]) / 2], [(b[1] + b[3]) / 2],
                                           crs=config.CRS)
            fold = folds.label(folds.station_km(centre))

        index.append(dict(name=name, scene=scene, col=col, row=row,
                          bounds=[round(v, 2) for v in b],
                          n_labels=int(labels.max()), n_inner=len(sel),
                          fold=str(fold),
                          corridor=round(float(share), 3),
                          valid=round(float((image > 0).mean()), 3)))
        if verbose and len(index) % 50 == 0:
            print(f"  {len(index)} tiles")

    if verbose:
        print(f"  {len(index)} tiles, {sum(t['n_inner'] for t in index)} boulders "
              f"in inner regions, corridor share "
              f"{np.mean([t['corridor'] for t in index]):.0%}")
    return index


def write_index(index, path=None):
    path = path or config.ARCHIVE / "index.json"
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def read_index(path=None):
    path = path or config.ARCHIVE / "index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def check_archive(index=None, out_dir=None):
    """
    Did the tiling actually work. Returns a list of complaints, empty is good.
    Worth running once, a silently half written archive would be painful later.
    """
    index = index if index is not None else read_index()
    out_dir = out_dir or config.ARCHIVE
    problems = []

    for t in index:
        for suffix in ("", "_masks", "_corridor"):
            f = out_dir / f"{t['name']}{suffix}.tif"
            if not f.exists():
                problems.append(f"missing: {f.name}")
            elif f.stat().st_size == 0:
                problems.append(f"empty: {f.name}")

    for scene in {t["scene"] for t in index}:
        tiled = sum(t["n_inner"] for t in index if t["scene"] == scene)
        mapped = len(geo.load_outlines(scene, min_d=config.MIN_TRAINING))
        if tiled > mapped:
            problems.append(f"{scene}: {tiled} tiled but only {mapped} mapped")
        elif tiled < mapped * 0.9:
            problems.append(f"{scene}: only {tiled} of {mapped} boulders landed "
                            f"in an inner region ({tiled / mapped:.0%})")

    for t in index:
        if t["n_labels"] and t["n_inner"] > t["n_labels"]:
            problems.append(f"{t['name']}: {t['n_inner']} inner > {t['n_labels']} burnt")

    return problems
