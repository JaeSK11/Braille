"""Turn raw 8x8 frames into the normalised 5x5 field h in [0,1].

Orientation flags live in orientation.json (written after the hand-wave test):
    {"flip_lr": false, "flip_ud": false, "transpose": false}
"""
import json, os
import numpy as np

INVALID = 4000
ORIENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orientation.json")

def load_orientation():
    if os.path.exists(ORIENT_FILE):
        return json.load(open(ORIENT_FILE))
    return {"flip_lr": False, "flip_ud": False, "transpose": False}

def orient(g, o=None):
    o = o or load_orientation()
    g = np.asarray(g)
    if o.get("transpose"): g = g.T
    if o.get("flip_lr"):   g = g[:, ::-1]
    if o.get("flip_ud"):   g = g[::-1, :]
    return g

def average_frames(frames):
    """frames: (N,8,8) ints. Mean over valid samples per zone; zones with no
    valid sample get the median of their valid neighbours. Returns (mean, std, nvalid)."""
    a = np.asarray(frames, dtype=float)
    valid = a < INVALID
    m = np.where(valid, a, np.nan)
    mean = np.nanmean(m, axis=0)
    std = np.nanstd(m, axis=0)
    nvalid = valid.sum(axis=0)
    # fill holes from neighbours
    holes = np.isnan(mean)
    if holes.any():
        filled = mean.copy()
        for r, c in zip(*np.where(holes)):
            nb = [mean[i, j] for i in range(max(0,r-1), min(8,r+2))
                              for j in range(max(0,c-1), min(8,c+2)) if not np.isnan(mean[i, j])]
            filled[r, c] = np.median(nb) if nb else np.nanmedian(mean)
        mean = filled
        std = np.where(holes, np.nan, std)
    return mean, std, nvalid

def crop5(g, r0=1, c0=1):
    return np.asarray(g)[r0:r0+5, c0:c0+5]

def auto_crop_offset(mean8):
    """Pick the 5x5 crop whose inner 3x3 is closest to the sensor (raised object centred).
    Offsets range 0..3. Returns (r0, c0)."""
    best = None
    for r0 in range(4):
        for c0 in range(4):
            inner = mean8[r0+1:r0+4, c0+1:c0+4].mean()
            if best is None or inner < best[0]:
                best = (inner, r0, c0)
    return best[1], best[2]

def depth_window(d5, lo_pct=5, hi_pct=95, margin=0.15):
    """Auto window around the scene: percentile span widened by `margin` each side."""
    lo, hi = np.percentile(d5, lo_pct), np.percentile(d5, hi_pct)
    span = max(hi - lo, 80.0)               # never narrower than 8 cm, so a flat wall stays flat
    return lo - margin*span, hi + margin*span

def normalise(d5, d_min=None, d_max=None):
    if d_min is None or d_max is None:
        d_min, d_max = depth_window(d5)
    h = (np.asarray(d5) - d_min) / (d_max - d_min)
    return np.clip(h, 0.1, 0.9), (d_min, d_max)

def otsu_threshold(vals):
    """Two-class threshold on a small set of depths (Otsu). Returns the split value."""
    v = np.sort(np.asarray(vals, float).ravel())
    best, thr = -1.0, v.mean()
    for i in range(1, len(v)):
        a, b = v[:i], v[i:]
        w = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if w > best:
            best, thr = w, (v[i-1] + v[i]) / 2
    return thr

def detrend(d5, min_step=30.0, iters=4, seed_frac=0.36):
    """Remove wall tilt. The wall is always the far surface, so seed a plane fit with the
    farthest ~36% of cells (9 of 25), then grow the inlier set to every cell within
    min_step/2 of that plane and refit. Works whether the object is 1 cell or 16.
    Only applied when the fitted wall ramps by more than min_step/2 across the crop;
    a level wall is returned unchanged. Returns depths with the wall flattened."""
    d5 = np.asarray(d5, float)
    rr, cc = np.mgrid[0:5, 0:5]
    A = np.column_stack([np.ones(25), rr.ravel(), cc.ravel()])
    flat = d5.ravel()
    k = max(6, int(round(seed_frac * 25)))
    far = np.zeros(25, bool); far[np.argsort(flat)[-k:]] = True
    plane = np.full((5, 5), flat.mean())
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[far], flat[far], rcond=None)
        plane = (A @ coef).reshape(5, 5)
        new_far = (np.abs(d5 - plane) < min_step / 2).ravel()
        if new_far.sum() < 6:
            break
        if np.array_equal(new_far, far):
            break
        far = new_far
    ramp = plane.max() - plane.min()
    if ramp < min_step / 2:          # wall is level to within the noise: leave the data alone
        return d5
    return d5 - plane + plane.mean()

def plateau(mean8, r0, c0, min_step=30.0):
    """True when the 5x5 crop is uniformly nearer than the 8x8 ring around it:
    a raised block that fills the whole window (no edges inside the 5x5 to see)."""
    m = np.asarray(mean8, float)
    crop = m[r0:r0+5, c0:c0+5]
    ring = np.ones_like(m, bool); ring[r0:r0+5, c0:c0+5] = False
    if not ring.any():
        return False
    near, far = np.median(crop), np.median(m[ring])
    return (far - near) > min_step and (np.abs(crop - near) < min_step/2).mean() >= 0.8

def _rectangularity(mask):
    """area / bounding-box area of the True cells; 0 if fewer than 2 cells."""
    rr, cc = np.where(mask)
    if len(rr) < 2:
        return 0.0
    bbox = (rr.max()-rr.min()+1) * (cc.max()-cc.min()+1)
    return len(rr) / bbox

def binarise(d5, near=0.15, far=0.85, min_step=30.0, quality=None):
    """Snap each cell to near/far. Half-covered zones are ambiguous, so try every
    candidate split between the two surface levels (Otsu, the midpoint, and each gap
    in the sorted depths) and keep the one whose near-mask is most rectangular; ties
    go to the larger mask (a half-covered zone belongs to the object as often as not),
    then to the larger depth gap. Flat scene (levels closer than
    min_step mm): return all-mid."""
    d5 = np.asarray(d5, float)
    lo, hi = float(d5.min()), np.percentile(d5, 90)
    if hi - lo < min_step:
        return np.full_like(d5, (near + far) / 2), (lo + hi) / 2
    v = np.sort(d5.ravel())
    cands = {otsu_threshold(d5), (lo + hi) / 2}
    for i in range(1, len(v)):
        if lo <= v[i-1] and v[i] <= hi + 1e-9 and v[i] - v[i-1] > 5.0:
            cands.add((v[i-1] + v[i]) / 2)
    best = None
    for thr in cands:
        mask = d5 <= thr
        n = mask.sum()
        if n < 1 or n > 20:
            continue
        gap = d5[~mask].min() - d5[mask].max()
        if gap < 10.0:                      # split inside the noise, not between surfaces
            continue
        q = quality(mask) if quality else _rectangularity(mask)
        key = (round(q, 2), int(n), gap)   # best-shaped mask, then largest, then cleanest
        if best is None or key > best[0]:
            best = (key, thr)
    thr = best[1] if best else (lo + hi) / 2
    return np.where(d5 <= thr, near, far), thr

def size_estimate(d5, shape):
    """(height_mm, width_mm) of a detected block of `shape` zones, using the zone width
    at the near surface. Uncertainty is about one zone per axis."""
    near = float(np.percentile(d5, 10))
    zone = 2 * near * np.tan(np.radians(30.0)) / 8
    return shape[0] * zone, shape[1] * zone, zone

def frames_to_h(frames, r0=1, c0=1, d_min=None, d_max=None, auto=False, binary=False, quality=None):
    mean, std, nvalid = average_frames(frames)
    mean = orient(mean)
    if auto:
        r0, c0 = auto_crop_offset(mean)
    d5 = crop5(mean, r0, c0)
    if binary:
        h, thr = binarise(detrend(d5), quality=quality)
        win = (thr, thr)
    else:
        h, win = normalise(d5, d_min, d_max)
    return h, d5, win, orient(std), (r0, c0)
