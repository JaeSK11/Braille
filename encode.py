"""25-qubit encoding of a 5x5 field and the readout circuits (plan section 6)."""
import math
from qiskit import QuantumCircuit

N = 5
def q(r, c): return N*r + c

EDGES = [((r,c),(r,c+1)) for r in range(N) for c in range(N-1)] + \
        [((r,c),(r+1,c)) for r in range(N-1) for c in range(N)]        # 40 edges

LAYERS = {
    "H_even": [((r,c),(r,c+1)) for r in range(N) for c in (0,2)],
    "H_odd":  [((r,c),(r,c+1)) for r in range(N) for c in (1,3)],
    "V_even": [((r,c),(r+1,c)) for c in range(N) for r in (0,2)],
    "V_odd":  [((r,c),(r+1,c)) for c in range(N) for r in (1,3)],
}
assert sorted(e for L in LAYERS.values() for e in L) == sorted(EDGES)

def encode(h):
    qc = QuantumCircuit(N*N, N*N)
    for r in range(N):
        for c in range(N):
            qc.ry(math.pi * float(h[r][c]), q(r,c))
    return qc

def bell_circuit(h, pairs):
    qc = encode(h)
    qc.barrier()
    for (a, b) in pairs:
        qa, qb = q(*a), q(*b)
        qc.cx(qa, qb); qc.h(qa)
        qc.measure([qa, qb], [qa, qb])
    return qc

def z_circuit(h):
    qc = encode(h); qc.barrier(); qc.measure(range(N*N), range(N*N)); return qc

def inverse_template_circuit(h, t):
    qc = encode(h); qc.barrier()
    for r in range(N):
        for c in range(N):
            qc.ry(-math.pi * float(t[r][c]), q(r,c))
    qc.measure(range(N*N), range(N*N)); return qc

def all_circuits(h):
    """dict name -> circuit: four Bell layers + one Z readout."""
    d = {name: bell_circuit(h, pairs) for name, pairs in LAYERS.items()}
    d["Z"] = z_circuit(h)
    return d
