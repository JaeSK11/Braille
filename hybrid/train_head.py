#!/usr/bin/env python3
"""Classical head on the cached feature maps (PLAN-hybrid-quantum-convolution.md, step 3; sections 4.3, 4.6).

Trains the same head on every feature set in a quanv_features.py cache and reports validation accuracy, so
rung 4a is read next to its controls. For the quantum sets (4a, C4) --shots also trains and evaluates at S
shots: the training features are resampled from the exact cache every epoch (the rung 4 noise curriculum),
the validation features are sampled once. 4a samples +-1 outcomes of <Z>; C4 samples the 11 outcome of P(11).

Heads (plan 4.3):
  cnn   two 3x3 convs (16 channels) -> global average pool -> linear      (~5k parameters)
  mlp   flatten -> 64 -> 64 -> linear                                       (the rung 2 shape)

Usage:
    python3 train_head.py data/quanv_n200_s0_j0.5_K4_L1_f0.npz --shots 0 50 200
"""
import argparse, time
import numpy as np
import torch
from torch import nn
import dataset, quanv

QUANTUM = {"4a", "C4"}

def head(kind, channels, n_classes):
    if kind == "cnn":
        return nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, n_classes))
    return nn.Sequential(nn.Flatten(), nn.Linear(channels * 49, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_classes))

def sample(name, F, shots, rng):
    if name == "4a":
        return quanv.sample_shots(F, shots, rng).astype(np.float32)
    return (rng.binomial(shots, np.clip(F, 0, 1)) / shots).astype(np.float32)        # C4: P(11)

def train(name, F, y, tr, va, kind, shots, epochs, lr, seed):
    """-> (best val acc, val acc at last epoch, n params)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = F[tr].mean((0, 2, 3), keepdims=True), F[tr].std((0, 2, 3), keepdims=True) + 1e-6
    norm = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    Fva = norm(sample(name, F[va], shots, rng) if shots else F[va])
    yt, yv = torch.tensor(y[tr]), torch.tensor(y[va])
    model = head(kind, F.shape[1], len(dataset.CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(); best = 0.0
    for ep in range(epochs):
        Ftr = norm(sample(name, F[tr], shots, rng) if shots else F[tr])
        model.train(); order = torch.randperm(len(tr))
        for i in range(0, len(tr), 64):
            idx = order[i:i + 64]
            opt.zero_grad(); loss_fn(model(Ftr[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Fva).argmax(1) == yv).float().mean().item()
        best = max(best, acc)
    return best, acc, sum(p.numel() for p in model.parameters())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", help="npz from quanv_features.py")
    ap.add_argument("--sets", default="4a,C1,C2,C2b,C4")
    ap.add_argument("--head", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--shots", type=int, nargs="+", default=[0], help="0 = exact; quantum sets only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.features); y, tr, va = d["y"], d["train"], d["val"]
    print(f"{a.features}: {len(tr)} train / {len(va)} val, head {a.head}, {a.epochs} epochs\n")
    print(f"{'set':5s} {'shots':>6s} {'params':>7s} {'best val':>9s} {'last val':>9s}")
    for name in a.sets.split(","):
        F = d[name]
        for S in (a.shots if name in QUANTUM else [0]):
            t0 = time.time()
            best, last, npar = train(name, F, y, tr, va, a.head, S, a.epochs, a.lr, a.seed)
            print(f"{name:5s} {S or 'exact':>6} {npar:7d} {best:9.3f} {last:9.3f}   {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
