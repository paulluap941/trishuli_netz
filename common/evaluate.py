"""
Scoring predicted masks against the hand drawn ones.

AP = TP / (TP + FN + FP), the metric ImageGrains uses. A prediction counts as
a hit when its IoU with a ground truth object is above the threshold, and each
ground truth object can only be hit once.

Sizes are scored too. A model can find every boulder and still get the
diameters wrong, which would ruin the grain size distribution.
"""
import numpy as np

import config


def areas(labels):
    """pixel count per id, index 0 is background"""
    return np.bincount(labels.ravel())


def iou_matrix(pred, true):
    """intersection over union for every predicted / true pair"""
    np_, nt = int(pred.max()), int(true.max())
    if np_ == 0 or nt == 0:
        return np.zeros((np_, nt))
    both = pred.astype(np.int64) * (nt + 1) + true.astype(np.int64)
    hist = np.bincount(both.ravel(), minlength=(np_ + 1) * (nt + 1))
    inter = hist.reshape(np_ + 1, nt + 1)[1:, 1:]
    ap, at = areas(pred)[1:], areas(true)[1:]
    union = ap[:, None] + at[None, :] - inter
    return inter / np.maximum(union, 1)


def match(pred, true, threshold=0.5):
    """greedy one to one matching, returns matched pairs and the counts"""
    iou = iou_matrix(pred, true)
    pairs = []
    if iou.size:
        work = iou.copy()
        while True:
            i, j = np.unravel_index(work.argmax(), work.shape)
            if work[i, j] < threshold:
                break
            pairs.append((i + 1, j + 1, float(work[i, j])))
            work[i, :] = 0
            work[:, j] = 0
    tp = len(pairs)
    return pairs, tp, int(pred.max()) - tp, int(true.max()) - tp


def diameter_m(labels, res):
    """equivalent circle diameter per id, in metres"""
    a = areas(labels)[1:] * res * res
    return 2 * np.sqrt(a / np.pi)


def score(pred, true, res=None, threshold=0.5, min_d=None):
    """one tile. drops objects below the size limit on both sides first."""
    res = res or config.RES
    min_d = min_d or config.MIN_DIAMETER
    pred, true = drop_small(pred, res, min_d), drop_small(true, res, min_d)
    pairs, tp, fp, fn = match(pred, true, threshold)

    dp, dt = diameter_m(pred, res), diameter_m(true, res)
    matched = [(dp[i - 1], dt[j - 1]) for i, j, _ in pairs]
    return dict(tp=tp, fp=fp, fn=fn,
                iou=[p[2] for p in pairs],
                d_pred=[m[0] for m in matched],
                d_true=[m[1] for m in matched],
                d_pred_all=list(dp), d_true_all=list(dt))


def restrict(labels, mask):
    """keep only objects whose centre of mass lies in mask"""
    n = int(labels.max())
    if n == 0:
        return labels
    keep = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) and mask[int(ys.mean()), int(xs.mean())]:
            keep.append(i)
    lut = np.zeros(n + 1, labels.dtype)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=labels.dtype)
    return lut[labels]


def inner_mask(shape, overlap=None):
    """true inside the part of a tile that is not shared with its neighbours"""
    overlap = overlap or config.OVERLAP
    m = np.zeros(shape, bool)
    h = overlap // 2
    m[h:shape[0] - h, h:shape[1] - h] = True
    return m


def drop_small(labels, res, min_d):
    d = diameter_m(labels, res)
    keep = np.nonzero(d >= min_d)[0] + 1
    if len(keep) == len(d):
        return labels
    lut = np.zeros(int(labels.max()) + 1, labels.dtype)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=labels.dtype)
    return lut[labels]


def summarise(scores):
    """pool per tile results into the numbers that go in the report"""
    tp = sum(s["tp"] for s in scores)
    fp = sum(s["fp"] for s in scores)
    fn = sum(s["fn"] for s in scores)
    dp = np.array([x for s in scores for x in s["d_pred"]])
    dt = np.array([x for s in scores for x in s["d_true"]])
    allp = np.array([x for s in scores for x in s["d_pred_all"]])
    allt = np.array([x for s in scores for x in s["d_true_all"]])

    out = dict(
        tp=tp, fp=fp, fn=fn,
        ap=tp / max(tp + fp + fn, 1),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        mean_iou=float(np.mean([x for s in scores for x in s["iou"]])) if tp else 0.0,
    )
    for name, arr in (("pred", allp), ("true", allt)):
        out[f"d50_{name}"] = float(np.median(arr)) if len(arr) else 0.0
        out[f"d84_{name}"] = float(np.percentile(arr, 84)) if len(arr) else 0.0
    if tp:
        out["size_bias"] = float(np.median(dp - dt))
        out["size_mae"] = float(np.median(np.abs(dp - dt)))
    else:
        out["size_bias"] = out["size_mae"] = 0.0
    return out
