"""Run the readout circuits on Aer with a simple noise model; return edge P(11) and Z P(1)."""
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit import transpile
import encode as E

def noise_model(p_read=0.02, p_cx=0.01):
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ReadoutError([[1-p_read, p_read],[p_read, 1-p_read]]))
    nm.add_all_qubit_quantum_error(depolarizing_error(p_cx, 2), ["cx"])
    return nm

def run(h, shots=100, noisy=True, p_read=0.02, p_cx=0.01, seed=None):
    sim = AerSimulator(method="matrix_product_state", noise_model=noise_model(p_read, p_cx) if noisy else None, seed_simulator=seed)
    circs = E.all_circuits(h)
    names = list(circs)
    tc = transpile([circs[n] for n in names], sim)
    res = sim.run(tc, shots=shots).result()
    edge_p11, z_p1 = {}, {}
    for name, c in zip(names, tc):
        counts = res.get_counts(c)
        if name == "Z":
            tot = sum(counts.values())
            for i in range(E.N*E.N):
                z_p1[i] = sum(v for k, v in counts.items() if k[::-1][i] == "1") / tot
        else:
            for (a, b) in E.LAYERS[name]:
                qa, qb = E.q(*a), E.q(*b)
                tot = sum(counts.values())
                n11 = sum(v for k, v in counts.items() if k[::-1][qa] == "1" and k[::-1][qb] == "1")
                edge_p11[(a, b)] = n11 / tot
    return edge_p11, z_p1
