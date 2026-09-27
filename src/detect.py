"""Square detection from the 40 edge values (see docs/method.md)."""
import encode as E

SHAPES = [(3,3), (3,4), (4,3), (2,3), (3,2), (2,2), (4,4), (2,4), (4,2)]   # (rows, cols)

def block_templates(shapes=SHAPES):
    """key = (rows, cols, r0, c0) -> set of boundary edges of that block inside the 5x5."""
    T = {}
    for (hgt, wid) in shapes:
        for r0 in range(E.N - hgt + 1):
            for c0 in range(E.N - wid + 1):
                inside = {(r,c) for r in range(r0, r0+hgt) for c in range(c0, c0+wid)}
                T[(hgt, wid, r0, c0)] = {e for e in E.EDGES if (e[0] in inside) != (e[1] in inside)}
    return T
TEMPL = block_templates()
SQUARE_ONLY = {k: v for k, v in TEMPL.items() if k[0] == 3 and k[1] == 3}

EDGE_ON = 0.06   # P(11) above this counts as "an edge is present"

def score_all(edge_p11, templates=None):
    out = {}
    for pos, bnd in (templates or TEMPL).items():
        rest = [e for e in E.EDGES if e not in bnd]
        b = sum(edge_p11[e] for e in bnd) / len(bnd)
        f = sum(edge_p11[e] for e in rest) / len(rest)
        cov_b = sum(edge_p11[e] > EDGE_ON for e in bnd) / len(bnd)
        cov_f = sum(edge_p11[e] > EDGE_ON for e in rest) / len(rest)
        out[pos] = (b - f, b, f, cov_b - cov_f)
    return out

def detect(edge_p11, z_p1, threshold=0.15, min_coverage=0.6, squares_only=False):
    scores = score_all(edge_p11, SQUARE_ONLY if squares_only else None)
    key, (s, b, f, cov) = max(scores.items(), key=lambda kv: kv[1][0] + kv[1][3])
    hgt, wid, r0, c0 = key
    pos = (r0, c0)
    found = s >= threshold and cov >= min_coverage
    sign = None
    if found:
        cells = [(r, c) for r in range(r0, r0+hgt) for c in range(c0, c0+wid)]
        inner = sum(z_p1[E.q(r, c)] for (r, c) in cells) / len(cells)
        ring = [(r, c) for r in range(E.N) for c in range(E.N) if (r, c) not in cells]
        outer = sum(z_p1[E.q(r, c)] for (r, c) in ring) / len(ring)
        sign = "raised" if inner < outer else "recessed"   # h is depth: smaller = closer = raised
    return {"found": found, "pos": pos, "shape": (hgt, wid), "score": s, "boundary_p11": b,
            "flat_p11": f, "coverage": cov, "sign": sign,
            "all_scores": {k: v[0] for k, v in scores.items()}}

def edge_grid_str(edge_p11):
    """ASCII picture: cells with horizontal/vertical edge strengths between them."""
    rows = []
    for r in range(E.N):
        line = ""
        for c in range(E.N):
            line += " o "
            if c < E.N-1:
                v = edge_p11[((r,c),(r,c+1))]
                line += "===" if v > 0.2 else ("---" if v > 0.08 else "   ")
        rows.append(line)
        if r < E.N-1:
            line = ""
            for c in range(E.N):
                v = edge_p11[((r,c),(r+1,c))]
                line += " H " if v > 0.2 else (" | " if v > 0.08 else "   ")
                line += "   "
            rows.append(line)
    return "\n".join(rows)
