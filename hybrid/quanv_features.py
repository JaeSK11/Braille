#!/usr/bin/env python3
"""Compute and cache rung 4a feature maps and the classical controls (PLAN-hybrid-quantum-convolution.md, step 2).

For every scene in the synthetic dataset, over 2x2 patches at stride 1 (7x7 map):
  4a   K fixed random quantum filters, <Z> per qubit           (N, 4K, 7, 7)
  C1   the raw patch values, no filter                        (N, 4,  7, 7)
  C2   random Fourier features over the 81 trig products      (N, 4K, 7, 7)
  C2b  random linear map + tanh                               (N, 4K, 7, 7)
  C4   the fixed Bell-edge readout P(11) on the 4 edges inside each patch (N, 4, 7, 7)
Optional shot sampling of 4a (--shots S). Saved to hybrid/data/quanv_*.npz with labels and the train/val split.

--probe fits a ridge classifier on each flattened feature set and prints train / val accuracy. That is a
linear read of the feature map, not the CNN head of step 3, so it is a floor, and it is the same floor for
every set.

Usage:
    python3 quanv_features.py --n-per-class 200 --filters 4 --probe
"""
import argparse, os, time
import numpy as np
import dataset, quanv

def all_patches(X, k=2):
    """(N,8,8) -> (N,7,7,k,k) stride-1 windows."""
    return np.lib.stride_tricks.sliding_window_view(X, (k, k), axis=(1, 2))

def feature_sets(X, filters, layers, reupload, seed, shots=None):
    k, n = 2, 4
    P = all_patches(X, k)                                               # (N,7,7,2,2)
    rng = np.random.default_rng(seed)
    thetas = [quanv.random_filter(k, layers, rng) for _ in range(filters)]
    q = np.concatenate([quanv.filter_features(P, th, k, reupload) for th in thetas], axis=-1)
    if shots:
        q = quanv.sample_shots(q, shots, rng)
    W2 = np.concatenate([quanv.random_fourier_weights(k, n, rng) for _ in range(filters)], axis=1)
    Wb, bb = rng.normal(size=(n, n * filters)) / np.sqrt(n), rng.normal(size=n * filters)
    flat = P.reshape(*P.shape[:3], n)
    bell = np.stack([np.sin(np.pi * (flat[..., b] - flat[..., a]) / 2) ** 2 / 2 for a, b in quanv.patch_edges(k)], -1)
    sets = {"4a": q, "C1": flat, "C2": quanv.random_fourier(P, W2), "C2b": quanv.random_tanh(P, Wb, bb), "C4": bell}
    return {name: np.moveaxis(F, -1, 1).astype(np.float32) for name, F in sets.items()}   # (N, C, 7, 7)

def ridge_probe(F, y, tr, va, n_classes, lam=1.0):
    """One-vs-rest ridge regression on the flattened map, standardised on train. -> (train acc, val acc)."""
    Z = F.reshape(len(F), -1).astype(float)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    Z = np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))])
    Y = np.eye(n_classes)[y]
    A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z[tr].T @ Y[tr])
    pred = (Z @ W).argmax(1)
    return (pred[tr] == y[tr]).mean(), (pred[va] == y[va]).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="dataset seed")
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--filters", type=int, default=4, help="K random quantum filters")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--reupload", action="store_true")
    ap.add_argument("--shots", type=int, default=0, help="sample 4a at this many shots (0 = exact)")
    ap.add_argument("--filter-seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    X, y, seeds = dataset.load(a.n_per_class, a.seed, jitter=a.jitter)
    tr, va = dataset.split(seeds)
    print(f"{len(X)} scenes, {len(tr)} train / {len(va)} val  ({time.time()-t0:.0f}s)")

    t1 = time.time()
    sets = feature_sets(X, a.filters, a.layers, a.reupload, a.filter_seed, a.shots or None)
    out = f"{dataset.DATA}/quanv_n{a.n_per_class}_s{a.seed}_j{a.jitter}_K{a.filters}_L{a.layers}" \
          f"{'_reup' if a.reupload else ''}{f'_S{a.shots}' if a.shots else ''}_f{a.filter_seed}.npz"
    os.makedirs(dataset.DATA, exist_ok=True)
    np.savez_compressed(out, y=y, train=tr, val=va, **sets)
    print("features " + ", ".join(f"{k} {v.shape[1:]}" for k, v in sets.items()) + f"  ({time.time()-t1:.0f}s)")
    print(f"saved {out}")

    if a.probe:
        print(f"\nridge probe, {len(dataset.CLASSES)} classes:")
        for name, F in sets.items():
            trn, val = ridge_probe(F, y, tr, va, len(dataset.CLASSES))
            print(f"  {name:4s} train {trn:.3f}  val {val:.3f}")

if __name__ == "__main__":
    main()
