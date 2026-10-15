"""
Tile and label generation.

Each scene is tiled in ITS OWN coordinate system. The images are never warped
onto a common grid - only resampled to the target resolution. That way the
labels sit exactly where they were drawn, and we interpolate once instead of
twice. The only thing moved between systems is the corridor polygon.

Per tile we write:
    <scene>_<col>_<row>.tif           image, uint16, raw values
    <scene>_<col>_<row>_masks.tif     label raster, uint16, one id per boulder
    <scene>_<col>_<row>_corridor.tif  uint8, 1 where the corridor was inspected

The corridor mask matters: only ~35 % of a tile lies inside the mapped strip.
Everything outside was never looked at, so it is neither labelled nor a valid
negative. Training must restrict crops to the corridor.

Normalisation is NOT baked in - it happens when loading, so it stays a model
parameter and can be applied identically to Vantor later.
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
from common import folds


def corridor_in_scene(scene):
    """corridor polygon, moved from november coordinates into the scene's own"""
    geom = gpd.read_file(config.corridor(scene)).to_crs(config.CRS).union_all()
    if scene == config.REFERENCE:
        return geom
    return Offset(config.VALLEY_AXIS, scene).nov_to_scene(geom)


def grid(bounds, res=None, tile=None, overlap=None):
    """tile origins on a fixed grid, stepping by the inner width"""
    res = res or config.RES
    tile = tile or config.TILE
    overlap = overlap or config.OVERLAP
    step = (tile - overlap) * res            # 192 m
    size = tile * res                        # 256 m
    pad = overlap * res / 2                  # half the border, so the inner
    x0, y0, x1, y1 = bounds                  # regions cover the corridor
    xs = np.arange(np.floor((x0 - pad) / step) * step, x1 + pad, step)
    ys = np.arange(np.floor((y0 - pad) / step) * step, y1 + pad, step)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            yield i, j, (x, y, x + size, y + size)


def read_window(src_path, bounds, size, res):
    """read a window and resample it to size x size at the target resolution"""
    with rasterio.open(src_path) as r:
        win = from_bounds(*bounds, transform=r.transform)
        a = r.read(1, window=win, out_shape=(size, size),
                   resampling=Resampling.bilinear,
                   boundless=True, fill_value=0)
    return a


def label_window(gdf, bounds, size, res):
    """burn outlines into a label raster, one integer per boulder"""
    tr = from_origin(bounds[0], bounds[3], res, res)
    hit = gdf.iloc[list(gdf.sindex.query(box(*bounds), predicate="intersects"))]
    if len(hit) == 0:
        return np.zeros((size, size), np.uint16), 0
    shapes = ((g, i) for i, g in enumerate(hit.geometry.values, start=1))
    out = rasterize(shapes, out_shape=(size, size), transform=tr,
                    fill=0, dtype=np.uint16)
    return out, int(out.max())


def build_scene(scene, out_dir=None, res=None, tile=None, overlap=None,
                min_diameter=None, verbose=True):
    """write all tiles of one scene, return the index as a list of dicts"""
    out_dir = out_dir or config.ARCHIVE
    res = res or config.RES
    tile = tile or config.TILE
    overlap = overlap or config.OVERLAP
    min_d = min_diameter if min_diameter is not None else config.MIN_TRAINING
    out_dir.mkdir(parents=True, exist_ok=True)

    img = config.image(scene)
    corr = corridor_in_scene(scene)
    prepare(corr)

    own = gpd.read_file(config.OUTLINES / f"{scene}_bloecke.gpkg").to_crs(config.CRS)
    nov = gpd.read_file(config.outlines(scene)).to_crs(config.CRS)
    d = 2 * np.sqrt(own.area / np.pi)
    keep = d >= min_d
    own, nov = own[keep].reset_index(drop=True), nov[keep].reset_index(drop=True)
    own["fold"] = folds.label(folds.station_km(nov.geometry))
    own["d_m"] = d[keep].values
    own.sindex                                     # build the index once

    index = []
    area = (tile * res) ** 2
    for i, j, b in grid(corr.bounds, res, tile, overlap):
        bx = box(*b)
        if not intersects(corr, bx):
            continue
        share = corr.intersection(bx).area / area
        if share < config.MIN_CORRIDOR:            # barely touches the strip
            continue
        a = read_window(img, b, tile, res)
        if (a > 0).mean() < 0.5:                   # mostly outside the image
            continue
        lab, n = label_window(own, b, tile, res)

        name = f"{scene}_{i:03d}_{j:03d}"
        tr = from_origin(b[0], b[3], res, res)
        prof = dict(driver="GTiff", height=tile, width=tile, count=1,
                    crs=config.CRS, transform=tr, compress="deflate",
                    tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out_dir / f"{name}.tif", "w", dtype="uint16", **prof) as w:
            w.write(a.astype(np.uint16), 1)
        with rasterio.open(out_dir / f"{name}_masks.tif", "w", dtype="uint16", **prof) as w:
            w.write(lab, 1)
        # valid = inside the corridor OR belonging to a labelled boulder.
        # the second part means a boulder straddling the corridor edge is
        # never partly ignored - nothing is ever cut off.
        mask = rasterize([(corr.intersection(bx), 1)], out_shape=(tile, tile),
                         transform=tr, fill=0, dtype=np.uint8)
        mask |= (lab > 0).astype(np.uint8)
        with rasterio.open(out_dir / f"{name}_corridor.tif", "w", dtype="uint8", **prof) as w:
            w.write(mask, 1)

        # fold of the tile: majority of its boulders, else by its centre
        inner = box(b[0] + overlap * res / 2, b[1] + overlap * res / 2,
                    b[2] - overlap * res / 2, b[3] - overlap * res / 2)
        sel = own.iloc[list(own.sindex.query(inner, predicate="intersects"))]
        sel = sel[sel.geometry.centroid.within(inner)]
        if len(sel):
            fold = sel["fold"].mode().iloc[0]
        else:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            fold = folds.label(folds.station_km(
                gpd.GeoSeries.from_xy([cx], [cy], crs=config.CRS)))

        index.append(dict(name=name, scene=scene, col=i, row=j,
                          bounds=[round(v, 2) for v in b],
                          n_labels=n, n_inner=int(len(sel)), fold=str(fold),
                          corridor=round(float(share), 3),
                          valid=round(float((a > 0).mean()), 3)))
        if verbose and len(index) % 50 == 0:
            print(f"  {scene}: {len(index)} tiles")

    if verbose:
        print(f"  {scene}: {len(index)} tiles, "
              f"{sum(t['n_inner'] for t in index)} boulders in inner regions, "
              f"corridor share {np.mean([t['corridor'] for t in index]):.0%}")
    return index


def write_index(index, path=None):
    path = path or (config.ARCHIVE / "index.json")
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path


def read_index(path=None):
    path = path or (config.ARCHIVE / "index.json")
    return json.loads(path.read_text(encoding="utf-8"))
