"""Identify which library shape is in the 5x5 field, from the 40 Bell-edge values and 25 Z reads.

For every template (and every in-grid translation of it) the expected edge set is the set
of 5x5 edges whose two cells differ in the mask. Score = mean P(11) on those edges minus
mean elsewhere; coverage = fraction of those edges lit minus fraction lit elsewhere.
A mask and its complement have the same edge set, so the Z readout decides between them:
the side with the lower mean P(1) (nearer) is the raised side.
"""
import numpy as np
import encode as E, shapes

def translations(mask):
    """All shifts of the mask's bounding box that stay inside 5x5, as (dr, dc, mask)."""
    rr, cc = np.where(mask)
    if len(rr) == 0:
        return []
    h = rr.max() - rr.min() + 1; w = cc.max() - cc.min() + 1
    base = mask[rr.min():rr.min()+h, cc.min():cc.min()+w]
    out = []
    for dr in range(5 - h + 1):
        for dc in range(5 - w + 1):
            m = np.zeros((5, 5), bool); m[dr:dr+h, dc:dc+w] = base
            out.append((dr, dc, m))
    return out

def edge_set(mask):
    return {e for e in E.EDGES if mask[e[0]] != mask[e[1]]}

_TEMPLATES = None
def templates(library=None):
    global _TEMPLATES
    if _TEMPLATES is None or library is not None:
        lib = library or shapes.LIBRARY
        T = []
        seen = {}
        for name, mask in lib.items():
            for dr, dc, m in translations(mask):
                es = frozenset(edge_set(m))
                if not es:                       # filled_all has no internal edges
                    T.append((name, dr, dc, m, es)); continue
                T.append((name, dr, dc, m, es))
        _TEMPLATES = T
    return _TEMPLATES

EDGE_ON = 0.06

def identify(edge_p11, z_p1, min_score=0.12, min_coverage=0.5):
    best = None
    lit = {e: edge_p11[e] > EDGE_ON for e in E.EDGES}
    zgrid = np.array([[z_p1[E.q(r, c)] for c in range(5)] for r in range(5)])
    ztot = zgrid.var() + 1e-9
    for name, dr, dc, m, es in templates():
        if not es:
            continue
        zin_v = zgrid[m]; zout_v = zgrid[~m]
        within = (zin_v.var() * len(zin_v) + zout_v.var() * len(zout_v)) / 25.0
        zsep = 1.0 - within / ztot            # 1 = mask splits Z into two tight groups
        rest = [e for e in E.EDGES if e not in es]
        b = np.mean([edge_p11[e] for e in es]); f = np.mean([edge_p11[e] for e in rest]) if rest else 0.0
        cb = np.mean([lit[e] for e in es]); cf = np.mean([lit[e] for e in rest]) if rest else 0.0
        score, cov = b - f, cb - cf
        # penalise templates whose boundary is a strict subset of what is lit (over-simple explanations)
        key = score + cov - 0.5 * cf + 0.5 * zsep
        if best is None or key > best[0]:
            best = (key, name, (dr, dc), m, score, cov)
    if best is None:
        return {"found": False}
    key, name, pos, m, score, cov = best
    found = score >= min_score and cov >= min_coverage
    # raised side from Z
    zin = np.mean([z_p1[E.q(r, c)] for r in range(5) for c in range(5) if m[r, c]])
    zout = np.mean([z_p1[E.q(r, c)] for r in range(5) for c in range(5) if not m[r, c]])
    raised_mask = zin < zout
    # if the raised side is the complement, report the complement's library name when it has one
    if found and not raised_mask:
        comp = ~m
        alt = next((n for n, mm in shapes.LIBRARY.items() if np.array_equal(mm, comp)), None)
        if alt:
            name, m, raised_mask = alt, comp, True
    return {"found": found, "name": name, "pos": pos, "mask": m, "score": float(score),
            "coverage": float(cov), "sign": "raised" if raised_mask else "recessed (complement raised)"}

def all_flat(edge_p11):
    return max(edge_p11.values()) < EDGE_ON


def library_quality(mask):
    """Best of: rectangularity, or intersection-over-union with any translated library shape
    (or its complement). Used to pick the snap threshold: a clean rectangle or a known shape."""
    import grid as _g
    best = _g._rectangularity(mask)
    for name, dr, dc, m, es in templates():
        for mm in (m, ~m):
            inter = np.logical_and(mask, mm).sum(); union = np.logical_or(mask, mm).sum()
            if union:
                best = max(best, inter / union)
    return best
