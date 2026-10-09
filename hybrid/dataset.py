#!/usr/bin/env python3
"""Synthetic 8x8 scenes for the learned models (PLAN-learned-shape-identifier.md section 4.1).

Per scene: a library shape or bare wall, wall distance, stand-off, sub-zone jitter, wall tilt on both axes,
a random 5x5 placement inside the 8x8, and 5 to 30 averaged frames. Output is the continuous h field over
the full frame (quanv.frames_to_h8), not the 5x5 crop, so position is left for the model to handle.

Split: 90 / 10 train / validation by scene seed (plan 4.4). Scenes are cached under hybrid/data/ (gitignored).

Usage:
    python3 dataset.py --n-per-class 200        # generate, cache, print a summary
"""
import argparse, os, time, warnings
from multiprocessing import Pool
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import shapes, sensor_sim as S, quanv

CLASSES = list(shapes.LIBRARY) + ["none"]          # 18
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def scene(args):
    """(label, seed, wall range, max jitter) -> (8,8) float32 h field."""
    label, seed, wall_range, jitter = args
    rng = np.random.default_rng(seed)
    wall = rng.uniform(*wall_range)
    n = int(rng.integers(5, 31))
    tilt_x, tilt_y = rng.uniform(0, 0.15, 2)
    if CLASSES[label] == "none":
        fr = S.frames(wall, [], n=n, tilt_x=tilt_x, tilt_y=tilt_y, rng=seed)
    else:
        crop = tuple(int(v) for v in rng.integers(0, 4, 2))
        w, boxes = shapes.scene(shapes.LIBRARY[CLASSES[label]], wall, rng.uniform(40, 120), crop, jitter, rng)
        fr = S.frames(w, boxes, n=n, tilt_x=tilt_x, tilt_y=tilt_y, rng=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return quanv.frames_to_h8(fr)[0].astype(np.float32)

def make(n_per_class, seed=0, wall_range=(200.0, 450.0), jitter=0.5, workers=None):
    """-> X (N,8,8) float32, y (N,) int, seeds (N,). Scene seed is derived from (dataset seed, class, index)."""
    jobs = [(k, seed * 1_000_003 + k * 100_000 + i, wall_range, jitter)
            for k in range(len(CLASSES)) for i in range(n_per_class)]
    with Pool(workers) as p:
        X = p.map(scene, jobs, chunksize=32)
    return np.stack(X), np.array([j[0] for j in jobs]), np.array([j[1] for j in jobs])

def split(seeds, val_frac=0.1):
    """Train / validation index arrays, deterministic in the scene seed."""
    h = (np.asarray(seeds, np.uint64) * np.uint64(2654435761)) >> np.uint64(12)
    val = (h % np.uint64(1000)) < val_frac * 1000
    return np.where(~val)[0], np.where(val)[0]

def cache_path(n_per_class, seed, wall_range, jitter):
    return f"{DATA}/scenes_n{n_per_class}_s{seed}_w{int(wall_range[0])}-{int(wall_range[1])}_j{jitter}.npz"

def load(n_per_class, seed=0, wall_range=(200.0, 450.0), jitter=0.5, workers=None):
    """make() with an on-disk cache."""
    p = cache_path(n_per_class, seed, wall_range, jitter)
    if os.path.exists(p):
        d = np.load(p); return d["X"], d["y"], d["seeds"]
    X, y, seeds = make(n_per_class, seed, wall_range, jitter, workers)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, X=X, y=y, seeds=seeds)
    return X, y, seeds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--wall", type=float, nargs=2, default=(200.0, 450.0))
    a = ap.parse_args()
    t0 = time.time()
    X, y, seeds = load(a.n_per_class, a.seed, tuple(a.wall), a.jitter)
    tr, va = split(seeds)
    print(f"{len(X)} scenes, {len(CLASSES)} classes, {len(tr)} train / {len(va)} val, {time.time()-t0:.0f}s")
    print(f"h range {X.min():.2f}..{X.max():.2f}, cached at {cache_path(a.n_per_class, a.seed, tuple(a.wall), a.jitter)}")
