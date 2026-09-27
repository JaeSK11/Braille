"""5x5 depth grid on 25 qubits: RY(pi*h) per cell, Bell-pair edge readout,
inverse-template readout. Product states => exact per-pair simulation, no SDK needed."""
import math, random
random.seed(7)
N = 5
P_READ = 0.02   # readout flip per qubit
P_CX   = 0.01   # depolarizing on each CNOT (pair outcome randomized)

def idx(r, c): return r*N + c
EDGES = [((r,c),(r,c+1)) for r in range(N) for c in range(N-1)] + \
        [((r,c),(r+1,c)) for r in range(N-1) for c in range(N)]   # 40 edges

def bell_probs(ta, tb):   # outcome probs (00,01,10,11) after CX,H on RY(ta)|0>,RY(tb)|0>
    return [math.cos((ta-tb)/2)**2/2, math.sin((ta+tb)/2)**2/2,
            math.cos((ta+tb)/2)**2/2, math.sin((ta-tb)/2)**2/2]

def sample_edge(ta, tb, shots):
    p = bell_probs(ta, tb); n11 = 0
    for _ in range(shots):
        if random.random() < P_CX: o = random.randrange(4)
        else:
            u, o, acc = random.random(), 3, 0.0
            for k in range(4):
                acc += p[k]
                if u < acc: o = k; break
        a, b = o >> 1, o & 1
        if random.random() < P_READ: a ^= 1
        if random.random() < P_READ: b ^= 1
        n11 += (a & b)
    return n11 / shots

def edge_map(h, shots):
    return {e: sample_edge(math.pi*h[e[0][0]][e[0][1]], math.pi*h[e[1][0]][e[1][1]], shots) for e in EDGES}

def square_templates():
    """boundary-edge sets for a 3x3 block at each of the 9 positions"""
    T = {}
    for r0 in range(N-2):
        for c0 in range(N-2):
            inside = {(r,c) for r in range(r0,r0+3) for c in range(c0,c0+3)}
            T[(r0,c0)] = {e for e in EDGES if (e[0] in inside) != (e[1] in inside)}
    return T
TEMPL = square_templates()

def match(em):
    best = None
    for pos, bnd in TEMPL.items():
        b = sum(em[e] for e in bnd)/len(bnd)
        f = sum(em[e] for e in EDGES if e not in bnd)/(len(EDGES)-len(bnd))
        s = b - f
        if best is None or s > best[1]: best = (pos, s, b, f)
    return best

def inverse_template_ones(h, templ_h, shots):
    """apply RY(-pi*t) to each cell, measure, count ones; expected ~0 on a match"""
    tot = 0
    for r in range(N):
        for c in range(N):
            p1 = math.sin(math.pi*(h[r][c]-templ_h[r][c])/2)**2
            for _ in range(shots):
                bit = 1 if random.random() < p1 else 0
                if random.random() < P_READ: bit ^= 1
                tot += bit
    return tot/shots

def z_read(h, r, c, shots):
    p1 = math.sin(math.pi*h[r][c]/2)**2; n=0
    for _ in range(shots):
        bit = 1 if random.random() < p1 else 0
        if random.random() < P_READ: bit ^= 1
        n += bit
    return n/shots

def grid(fn): return [[fn(r,c) for c in range(N)] for r in range(N)]
IN = lambda r,c,r0=1,c0=1: r0<=r<r0+3 and c0<=c<c0+3
scenes = {
 "raised square (centre)":   grid(lambda r,c: 0.8 if IN(r,c) else 0.2),
 "recessed square (centre)": grid(lambda r,c: 0.2 if IN(r,c) else 0.8),
 "raised square (top-left)": grid(lambda r,c: 0.8 if IN(r,c,0,0) else 0.2),
 "flat plane":               grid(lambda r,c: 0.5),
 "tilted plane":             grid(lambda r,c: 0.2 + 0.15*c),
 "single raised pixel":      grid(lambda r,c: 0.8 if (r,c)==(2,2) else 0.2),
 "raised plus/cross":        grid(lambda r,c: 0.8 if (r==2 or c==2) else 0.2),
 "square + 5% noise":        grid(lambda r,c: (0.8 if IN(r,c) else 0.2) + random.gauss(0,0.05)),
}
CENTRE_T = scenes["raised square (centre)"]
for shots in (1000, 100):
    print(f"\n=== {shots} shots per circuit  (readout err {P_READ:.0%}/qubit, CNOT depol {P_CX:.0%}) ===")
    print(f"{'scene':26s} {'best sq. pos':12s} {'score':>6s} {'bnd P11':>8s} {'flat P11':>9s} {'invT ones':>10s} {'Z(2,2)':>7s} {'Z(0,0)':>7s}")
    for name, h in scenes.items():
        em = edge_map(h, shots)
        pos, s, b, f = match(em)
        ones = inverse_template_ones(h, CENTRE_T, shots)
        print(f"{name:26s} {str(pos):12s} {s:6.3f} {b:8.3f} {f:9.3f} {ones:10.2f} {z_read(h,2,2,shots):7.2f} {z_read(h,0,0,shots):7.2f}")
print("\nideal boundary P11 for a 0.6*pi step:", round((1-math.cos(0.6*math.pi)**2*0+ (1-math.cos(0.3*math.pi)**2))/2 - 0.5 + 0.5*(1-math.cos(0.3*math.pi)**2)*0, 3) if False else round((1-math.cos(0.3*math.pi)**2)/2, 3))
