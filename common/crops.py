"""
Cuts training crops out of the tile archive.

Cellpose-SAM is locked to 256 px crops, so the crops come out at 384 and
cellpose takes its 256 from inside them. That margin means a rotation never
pulls in empty corners.

Two kinds of crop:
    on a boulder    centred on an outline, with some jitter
    background      random position in the corridor

Only positions where the whole crop is inspected ground are used, so the
label raster is the full truth for every pixel the net sees.
"""
import numpy as np
import rasterio

import config

SIZE = 384
MIN_VALID = 0.98


def read_tile(name, archive=None):
    archive = archive or config.ARCHIVE
    out = []
    for suffix in ("", "_masks", "_corridor"):
        with rasterio.open(archive / f"{name}{suffix}.tif") as r:
            out.append(r.read(1))
    return out


def positions(labels, valid, size=SIZE, per_boulder=1, background=0.3,
              min_valid=MIN_VALID, rng=None):
    """top left offsets for the crops of one tile"""
    rng = rng or np.random.default_rng()
    h, w = valid.shape
    half = size // 2
    span = h - size
    if span <= 0:
        return []

    def ok(y, x):
        return valid[y:y + size, x:x + size].mean() >= min_valid

    out = []
    ids = np.unique(labels)
    ids = ids[ids > 0]
    for i in ids:
        ys, xs = np.nonzero(labels == i)
        cy, cx = int(ys.mean()), int(xs.mean())
        for _ in range(per_boulder):
            for _try in range(12):
                jy, jx = rng.integers(-half // 2, half // 2 + 1, 2)
                y = int(np.clip(cy - half + jy, 0, span))
                x = int(np.clip(cx - half + jx, 0, span))
                if ok(y, x):
                    out.append((y, x))
                    break

    for _ in range(int(round(len(ids) * background)) or 1):
        for _try in range(12):
            y, x = rng.integers(0, span + 1, 2)
            if ok(y, x):
                out.append((int(y), int(x)))
                break
    return out


def build(names, size=SIZE, per_boulder=1, background=0.3, min_valid=MIN_VALID,
          res_factor=1, archive=None, seed=0, verbose=True):
    """
    Crops for a list of tiles.

    res_factor > 1 downsamples, so the same crop covers more ground. 1 keeps
    the archive resolution of 0.125 m/px.
    """
    rng = np.random.default_rng(seed)
    imgs, lbls = [], []

    for k, name in enumerate(names):
        img, lab, valid = read_tile(name, archive)
        for y, x in positions(lab, valid, size, per_boulder, background,
                              min_valid, rng):
            a = img[y:y + size, x:x + size]
            b = lab[y:y + size, x:x + size]
            if res_factor > 1:
                a = a[::res_factor, ::res_factor]
                b = b[::res_factor, ::res_factor]
            # renumber so ids are 1..n inside the crop
            ids = np.unique(b)
            ids = ids[ids > 0]
            if len(ids):
                lut = np.zeros(int(b.max()) + 1, np.uint16)
                lut[ids] = np.arange(1, len(ids) + 1, dtype=np.uint16)
                b = lut[b]
            imgs.append(a)
            lbls.append(b)
        if verbose and (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(names)} tiles, {len(imgs)} crops")

    if verbose:
        n = sum(int(b.max()) for b in lbls)
        empty = sum(1 for b in lbls if b.max() == 0)
        print(f"  {len(imgs)} crops, {n} boulders, {empty / len(imgs):.0%} empty")
    return imgs, lbls
