#!/usr/bin/env python3
"""Rung 4b: quantum filters trained end-to-end with the CNN head (PLAN-hybrid-quantum-convolution.md, step 4).

The 2x2 filter of quanv.py written in torch: product-state RY(pi*h) encoding, L layers of [RY, RZ per qubit]
then the fixed nearest-neighbour CNOTs, optional re-uploading of the encoding before every layer after the
first, <Z> per qubit. Exact statevector, exact gradients by backprop through the 16-dim state. K filters give
a (B, 4K, 7, 7) map into the same head as train_head.py.

Modes:
  trained   filter angles and head trained together (rung 4b)
  frozen    filter angles fixed at their random draw, head trained (rung 4a inside this code, as a check)
  classical C3: a trained 2x2 conv layer with 4K channels + tanh, same head

--test also scores the held-out Test 1 split (plan 4.4 of the learned-identifier plan: wall 450 to 550 mm,
outside the training range) at the epoch with the best validation accuracy and at the last epoch. Epoch
selection never sees the test set.

Usage:
    python3 train_quanv.py --mode trained --layers 2 --reupload
    python3 train_quanv.py --mode classical --test --epochs 60 --lr 1e-2
"""
import argparse, time
import numpy as np
import torch
from torch import nn
import dataset, quanv, train_head

K2, N, D = 2, 4, 16                                   # kernel, qubits, state dim
CNOTS = torch.tensor(np.linalg.multi_dot([quanv.cnot(N, c, t) for c, t in quanv.patch_edges(K2)][::-1]), dtype=torch.complex128)
ZSIGN = torch.tensor(quanv._z_signs(N), dtype=torch.float64)

def ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex128)

def rz(t):
    z = torch.zeros_like(t)
    return torch.stack([torch.stack([torch.exp(-0.5j * t), z], -1), torch.stack([z, torch.exp(0.5j * t)], -1)], -2)

def apply_1q(psi, G, j):
    """psi (B, 16), G (B, 2, 2) or (2, 2): apply G to qubit j."""
    B = psi.shape[0]
    x = psi.reshape(B, *([2] * N)).movedim(j + 1, -1).reshape(B, D // 2, 2)
    x = torch.einsum('nab,nxb->nxa', G.expand(B, 2, 2), x)
    return x.reshape(B, *([2] * N)).movedim(-1, j + 1).reshape(B, D)

def encode(psi, h):
    """h (B, 4): RY(pi*h_j) on every qubit."""
    G = ry(np.pi * h)
    for j in range(N):
        psi = apply_1q(psi, G[:, j], j)
    return psi

class QuantumFilters(nn.Module):
    def __init__(self, filters, layers, reupload, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.rand(filters, layers, N, 2, generator=g, dtype=torch.float64) * 2 * np.pi)
        self.reupload = reupload

    def layer_unitary(self, th):
        """th (4, 2) -> (16, 16): kron of RZ·RY per qubit, then the CNOTs."""
        U = torch.ones(1, 1, dtype=torch.complex128)
        for j in range(N):
            U = torch.kron(U, rz(th[j, 1]) @ ry(th[j, 0]))
        return CNOTS @ U

    def forward(self, h):
        """h (B, 4) in [0, 1] -> (B, 4K) <Z> features."""
        B = h.shape[0]
        psi0 = torch.zeros(B, D, dtype=torch.complex128); psi0[:, 0] = 1
        psi0 = encode(psi0, h)
        out = []
        for th in self.theta:
            psi = psi0
            for l in range(th.shape[0]):
                if self.reupload and l > 0:
                    psi = encode(psi, h)
                psi = psi @ self.layer_unitary(th[l]).T
            out.append((psi.abs() ** 2) @ ZSIGN)
        return torch.cat(out, -1)

class Model(nn.Module):
    def __init__(self, mode, filters, layers, reupload, seed):
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.filt = nn.Conv2d(1, N * filters, K2)
        else:
            self.filt = QuantumFilters(filters, layers, reupload, seed)
            if mode == "frozen":
                self.filt.theta.requires_grad_(False)
        self.head = train_head.head("cnn", N * filters, len(dataset.CLASSES))

    def feature_map(self, X):
        """X (B, 8, 8) -> (B, 4K, 7, 7)."""
        if self.mode == "classical":
            return torch.tanh(self.filt(X[:, None].float()))
        P = X.unfold(1, K2, 1).unfold(2, K2, 1).reshape(len(X), 49, N)
        F = self.filt(P.reshape(-1, N).double()).reshape(len(X), 49, -1)
        return F.permute(0, 2, 1).reshape(len(X), -1, 7, 7).float()

    def forward(self, X):
        return self.head(self.feature_map(X))

def _test():
    """torch filter == numpy quanv.filter_features, both re-uploading modes."""
    rng = np.random.default_rng(0)
    P = rng.uniform(0.1, 0.9, (50, K2, K2))
    for reup in (False, True):
        qf = QuantumFilters(1, 2, reup, seed=3)
        want = quanv.filter_features(P, qf.theta[0].detach().numpy(), K2, reupload=reup)
        got = qf(torch.tensor(P.reshape(50, N))).detach().numpy()
        assert np.allclose(got, want, atol=1e-10), f"torch vs numpy, reupload={reup}"
    print("train_quanv self-test passed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trained", "frozen", "classical"], default="trained")
    ap.add_argument("--filters", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--reupload", action="store_true")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0, help="dataset seed")
    ap.add_argument("--filter-seed", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="also score Test 1 (wall 450-550 mm, 50 scenes per class)")
    a = ap.parse_args()
    _test()
    torch.manual_seed(a.filter_seed)

    X, y, seeds = dataset.load(a.n_per_class, a.seed, jitter=a.jitter)
    tr, va = dataset.split(seeds)
    X, y = torch.tensor(X, dtype=torch.float64), torch.tensor(y)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    if a.test:
        Xt, yt, _ = dataset.load(50, a.seed + 1, wall_range=(450.0, 550.0), jitter=a.jitter)
        Xt, yt = torch.tensor(Xt, dtype=torch.float64), torch.tensor(yt)
    model = Model(a.mode, a.filters, a.layers, a.reupload, a.filter_seed)
    nf = sum(p.numel() for p in model.filt.parameters() if p.requires_grad)
    nh = sum(p.numel() for p in model.head.parameters())
    tag = f"{a.mode} K={a.filters} L={a.layers}{' reupload' if a.reupload else ''}"
    print(f"{tag}: {nf} filter + {nh} head parameters, {len(tr)} train / {len(va)} val")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    loss_fn = nn.CrossEntropyLoss(); best = 0.0; test_at_best = test = float("nan"); t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); order = torch.randperm(len(tr)); tot = 0.0
        for i in range(0, len(tr), 64):
            idx = order[i:i + 64]
            opt.zero_grad(); loss = loss_fn(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
            if a.test:
                test = (model(Xt).argmax(1) == yt).float().mean().item()
        if acc > best:
            best, test_at_best = acc, test
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {tot/len(tr):.4f}  val {acc:.3f}  {time.time()-t0:.0f}s", flush=True)
    line = f"\n{tag}: best val {best:.3f}, last {acc:.3f}"
    if a.test:
        line += f", test1 at best-val epoch {test_at_best:.3f}, test1 last {test:.3f}"
    print(line)

if __name__ == "__main__":
    main()
