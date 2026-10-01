#!/usr/bin/env python3
"""Render every library shape with sensor_sim, push it through the 25-qubit pipeline, identify it."""
import sys, time
import numpy as np
import shapes, grid, run_sim, identify

def run(mask, jitter=0.0, seed=0, wall=300.0, depth=80.0, shots=100):
    fr = shapes.frames(mask, n=20, wall=wall, depth=depth, jitter=jitter, rng=seed)
    mean8 = grid.orient(grid.average_frames(fr)[0])
    if grid.plateau(mean8, 1, 1):
        return {"found": True, "name": "filled_all_25", "score": 0.0, "coverage": 0.0, "sign": "raised plateau (from 8x8 ring)"}
    h, d5, win, std5, off = grid.frames_to_h(fr, auto=False, binary=True, quality=identify.library_quality)
    e, z = run_sim.run(h, shots=shots, seed=seed)
    if identify.all_flat(e):
        return {"found": False, "name": "flat", "score": 0.0, "coverage": 0.0, "sign": "-"}
    return identify.identify(e, z)

def main():
    jitters = [0.0, 0.3]
    t0 = time.time()
    print(f"{'shape':20s} {'jitter':>6s} {'identified as':22s} {'score':>6s} {'cov':>5s}  sign")
    ok = tot = 0
    for name, mask in shapes.LIBRARY.items():
        for j in jitters:
            r = run(mask, jitter=j, seed=hash((name, j)) % 10000)
            got = r.get("name", "-") if r.get("found") or r.get("name") == "flat" else "no shape"
            good = got == name
            ok += good; tot += 1
            print(f"{name:20s} {j:6.1f} {got:22s} {r.get('score',0):6.3f} {r.get('coverage',0):5.2f}  {r.get('sign','-')}{'' if good else '   <-- expected '+name}")
    print(f"\n{ok}/{tot} identified correctly, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
