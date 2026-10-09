#!/usr/bin/env python3
"""Quantum convolution over the 8x8 range image, mode 1 (PLAN-hybrid-quantum-convolution.md, section 4).

Every kxk patch of the normalised field h is encoded as a product state, RY(pi*h) per cell as in encode.py,
run through a small fixed circuit, and read out as <Z> on each qubit. Exact statevector in numpy; no SDK.

  patch (k,k) -> n = k*k qubits, cell (r,c) is qubit k*r + c
  filter      -> L layers of [RY, RZ on every qubit] then CNOTs on patch nearest neighbours (plan 4.2)
  readout     -> <Z_i> for every qubit, optionally sampled at S shots (plan 4.6)
  re-upload   -> repeat the RY(pi*h) encoding before every layer after the first (plan 4.2)

Without re-uploading each <Z_i> is a linear combination of the 3^n products of {1, cos(pi h_j), sin(pi h_j)},
so the fixed random filter is a random projection of `trig_features` (plan 4.4). `random_fourier` is that
control (C2); `random_tanh` is the generic one (C2b).

Lives in hybrid/ (the hybrid approach); the circuit-only pipeline in src/ is imported, never edited.

Usage:
    python3 quanv.py          # self-tests, then one synthetic scene's feature map
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import grid

# ---------------------------------------------------------------- gates

def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

def one_qubit(n, i, g):
    """g on qubit i of n, identity elsewhere. Qubit 0 is the leftmost kron factor."""
    U = np.array([[1.0 + 0j]])
    for j in range(n):
        U = np.kron(U, g if j == i else np.eye(2))
    return U

def cnot(n, c, t):
    d = 2 ** n
    U = np.zeros((d, d))
    for b in range(d):
        cb, tb = (b >> (n - 1 - c)) & 1, n - 1 - t
        U[b ^ (cb << tb), b] = 1.0
    return U

def patch_edges(k):
    """Nearest-neighbour (control, target) pairs of a kxk patch, row-major qubit index. Same shape as encode.EDGES."""
    q = lambda r, c: k * r + c
    return [(q(r, c), q(r, c + 1)) for r in range(k) for c in range(k - 1)] + \
           [(q(r, c), q(r + 1, c)) for r in range(k - 1) for c in range(k)]

def layer_unitary(theta_l, k):
    """theta_l: (n, 2) RY, RZ angles. Rotations on every qubit, then the nearest-neighbour CNOTs."""
    n = k * k
    U = np.eye(2 ** n, dtype=complex)
    for i in range(n):
        U = one_qubit(n, i, rz(theta_l[i, 1]) @ ry(theta_l[i, 0])) @ U
    for c, t in patch_edges(k):
        U = cnot(n, c, t) @ U
    return U

def random_filter(k=2, layers=1, rng=None):
    """Fixed random filter angles, shape (layers, k*k, 2)."""
    rng = np.random.default_rng(rng)
    return rng.uniform(0, 2 * np.pi, (layers, k * k, 2))

# ---------------------------------------------------------------- encoding and readout

def patches(h, k=2, stride=1):
    """(8,8) field -> (H', W', k, k) sliding windows."""
    h = np.asarray(h, dtype=float)
    w = np.lib.stride_tricks.sliding_window_view(h, (k, k))
    return w[::stride, ::stride]

def _kron_rows(factors):
    """factors: (..., n, m) -> (..., m^n) kron product over the n axis."""
    out = factors[..., 0, :]
    for j in range(1, factors.shape[-2]):
        out = (out[..., :, None] * factors[..., j, :][..., None, :]).reshape(*out.shape[:-1], -1)
    return out

def encode_states(P):
    """P: (..., n) h values -> (..., 2^n) product state RY(pi*h)|0> per qubit."""
    a = np.pi * np.asarray(P) / 2
    return _kron_rows(np.stack([np.cos(a), np.sin(a)], axis=-1)).astype(complex)

def encoding_matrices(P):
    """P: (..., n) -> (..., 2^n, 2^n) kron of RY(pi*h_j), for re-uploading."""
    a = np.pi * np.asarray(P)
    c, s = np.cos(a / 2), np.sin(a / 2)
    G = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)      # (..., n, 2, 2)
    U = G[..., 0, :, :]
    for j in range(1, G.shape[-3]):
        U = np.einsum('...ab,...cd->...acbd', U, G[..., j, :, :]).reshape(*U.shape[:-2], U.shape[-2] * 2, U.shape[-1] * 2)
    return U.astype(complex)

def _z_signs(n):
    b = np.arange(2 ** n)
    return np.stack([1 - 2 * ((b >> (n - 1 - i)) & 1) for i in range(n)], axis=-1)   # (2^n, n)

def z_expect(states, n):
    """(..., 2^n) states -> (..., n) <Z_i>."""
    return (np.abs(states) ** 2) @ _z_signs(n)

def filter_features(P, theta, k=2, reupload=False):
    """P: (..., k, k) patches, theta: (L, k*k, 2) -> (..., k*k) exact <Z> features."""
    n = k * k
    flat = np.asarray(P).reshape(*np.shape(P)[:-2], n)
    psi = encode_states(flat)
    for l in range(theta.shape[0]):
        if reupload and l > 0:
            psi = np.einsum('...ij,...j->...i', encoding_matrices(flat), psi)
        psi = psi @ layer_unitary(theta[l], k).T
    return z_expect(psi, n)

def sample_shots(z, shots, rng=None):
    """Replace exact <Z> by the S-shot estimate: each shot is +-1 with P(+1) = (1+z)/2."""
    rng = np.random.default_rng(rng)
    return 2 * rng.binomial(shots, np.clip((1 + np.asarray(z)) / 2, 0, 1)) / shots - 1

def features(h, thetas, k=2, stride=1, reupload=False, shots=None, rng=None):
    """(8,8) field, list of K filters -> (K*k*k, H', W') feature map, channels first."""
    P = patches(h, k, stride)
    F = np.concatenate([filter_features(P, th, k, reupload) for th in thetas], axis=-1)
    if shots:
        F = sample_shots(F, shots, rng)
    return np.moveaxis(F, -1, 0)

# ---------------------------------------------------------------- classical controls (plan section 6)

def trig_features(P):
    """P: (..., n) -> (..., 3^n): every product of one of {1, cos(pi h_j), sin(pi h_j)} per cell."""
    a = np.pi * np.asarray(P)
    return _kron_rows(np.stack([np.ones_like(a), np.cos(a), np.sin(a)], axis=-1))

def random_fourier(P, W):
    """C2: random linear map over trig_features. W: (3^n, n_out), from random_fourier_weights."""
    n = np.shape(P)[-1] * np.shape(P)[-2]
    return trig_features(np.asarray(P).reshape(*np.shape(P)[:-2], n)) @ W

def random_fourier_weights(k=2, n_out=None, rng=None):
    rng = np.random.default_rng(rng)
    n = k * k
    return rng.normal(size=(3 ** n, n_out or n)) / np.sqrt(3 ** n)

def random_tanh(P, W, b):
    """C2b: tanh(patch @ W + b). W: (n, n_out), b: (n_out,)."""
    n = np.shape(P)[-1] * np.shape(P)[-2]
    return np.tanh(np.asarray(P).reshape(*np.shape(P)[:-2], n) @ W + b)

# ---------------------------------------------------------------- input (plan 4.1)

def frames_to_h8(frames, d_min=None, d_max=None):
    """(N,8,8) frames -> (8,8) continuous h in [0.1, 0.9] over the full frame, no crop. Returns (h, window)."""
    mean = grid.orient(grid.average_frames(frames)[0])
    return grid.normalise(mean, d_min, d_max)

# ---------------------------------------------------------------- self-tests

def _tests():
    rng = np.random.default_rng(0)
    k, n = 2, 4
    th = random_filter(k, layers=2, rng=1)

    # gates: cnot flips the target iff the control is set; layer unitary is unitary
    for b in range(16):
        e = np.zeros(16); e[b] = 1
        out = cnot(4, 1, 3) @ e
        assert out[b ^ (((b >> 2) & 1) << 0)] == 1, "cnot(4,1,3)"
    U = layer_unitary(th[0], k)
    assert np.allclose(U.conj().T @ U, np.eye(16)), "unitary"

    # no rotation, no entangler (k=1): <Z> = cos(pi h)
    h1 = rng.uniform(0.1, 0.9, (8, 8))
    F1 = filter_features(patches(h1, 1), np.zeros((1, 1, 2)), k=1)[..., 0]
    assert np.allclose(F1, np.cos(np.pi * h1)), "k=1 readout"

    # flat field: every patch gives the same feature vector
    F = features(np.full((8, 8), 0.3), [th], k)
    assert F.shape == (4, 7, 7) and np.allclose(F, F[:, :1, :1]), "flat"

    # step edge at column 3|4: only straddling patches (start column 3) differ from the flat values
    h = np.full((8, 8), 0.3); h[:, 4:] = 0.7
    Fs = features(h, [th], k)
    lo, hi = features(np.full((8, 8), 0.3), [th], k)[:, 0, 0], features(np.full((8, 8), 0.7), [th], k)[:, 0, 0]
    assert np.allclose(Fs[:, :, :3], lo[:, None, None]) and np.allclose(Fs[:, :, 4:], hi[:, None, None]), "step sides"
    assert not np.allclose(Fs[:, :, 3], lo[:, None]) and not np.allclose(Fs[:, :, 3], hi[:, None]), "step edge"

    # translation: shift the input one column, the map shifts one column
    h = rng.uniform(0.1, 0.9, (8, 8))
    Fa, Fb = features(h, [th], k), features(np.roll(h, 1, axis=1), [th], k)
    assert np.allclose(Fb[:, :, 1:], Fa[:, :, :-1]), "shift"

    # closed form (plan 4.4): fits the 81 trig products exactly without re-uploading, not with it
    P = rng.uniform(0.1, 0.9, (600, k, k))
    T = trig_features(P.reshape(600, n))
    for reup, want_exact in ((False, True), (True, False)):
        Q = filter_features(P, th, k, reupload=reup)
        resid = np.linalg.norm(T @ np.linalg.lstsq(T, Q, rcond=None)[0] - Q) / np.linalg.norm(Q)
        assert (resid < 1e-9) == want_exact, f"closed form reupload={reup} resid={resid:.2e}"

    # controls have the quantum output shape; shots stay in range and converge
    assert random_fourier(P, random_fourier_weights(k, rng=2)).shape == (600, 4)
    assert random_tanh(P, rng.normal(size=(4, 4)), np.zeros(4)).shape == (600, 4)
    S = sample_shots(Fa, 50, rng); assert S.shape == Fa.shape and np.all(np.abs(S) <= 1)
    assert np.abs(sample_shots(Fa, 200_000, rng) - Fa).max() < 0.02, "shots converge"
    print("quanv self-tests passed")

if __name__ == "__main__":
    _tests()
    import shapes
    fr = shapes.frames(shapes.LIBRARY["plus"], n=20, crop=(1, 1), rng=0)
    h, win = frames_to_h8(fr)
    F = features(h, [random_filter(2, layers=1, rng=0)], shots=None)
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print("h (8x8), window %.0f..%.0f mm" % win); print(h)
    print("<Z_0> feature map (7x7), one random 2x2 filter"); print(F[0])
