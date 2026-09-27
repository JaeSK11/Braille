#!/usr/bin/env python3
"""Synthetic sweep: scenes -> sensor_sim frames -> grid -> 25-qubit circuits -> Aer -> detect.
Reports whether the detected shape/position/sign match the truth."""
import sys, time, itertools
import numpy as np
import sensor_sim as S, grid, run_sim, detect, encode as E

def truth_block(wall, box):
    """Expected (rows, cols) in zones, from the face size at the face distance, rounded."""
    z = S.zone_mm(wall - box.depth)
    return (int(round(box.h / z)), int(round(box.w / z)))

def run_scene(fr, shots=100, seed=None):
    h, d5, win, std5, (r0, c0) = grid.frames_to_h(fr, auto=True, binary=True)
    e, z = run_sim.run(h, shots=shots, seed=seed)
    return detect.detect(e, z)

def main():
    rng = np.random.default_rng(0)
    rows = []
    t0 = time.time()
    # --- 1. blocks: target sizes in zones x distances x positions
    for zones in (2, 3, 4):
        for wall in (250, 300, 400):
            depth = 80
            z = S.zone_mm(wall - depth)
            side = zones * z * 1.05                     # slightly over N zones so N solid zones survive snapping
            for (ox, oy) in ((0,0), (-0.5,0), (0,0.5), (-1.0,-1.0), (0.5,-0.5)):
                box = S.Box(ox*z, oy*z, side, side, depth)
                fr = S.frames(wall, [box], n=20, tilt_x=0.03, rng=rng.integers(1e9))
                r = run_scene(fr, seed=int(rng.integers(1e9)))
                ok_shape = r["found"] and all(abs(a-b) <= 1 for a, b in zip(r["shape"], (zones, zones))) and r["sign"] == "raised"
                rows.append(("square", zones, wall, (ox,oy), r["found"], r["shape"], r["sign"], round(r["score"],3), round(r["coverage"],2), ok_shape))
    # --- 2. rectangles
    for (hz, wz) in ((3,4), (4,3), (2,3)):
        wall, depth = 300, 80; z = S.zone_mm(wall - depth)
        box = S.Box(0, 0, wz*z*1.05, hz*z*1.05, depth)
        fr = S.frames(wall, [box], n=20, rng=rng.integers(1e9)); r = run_scene(fr, seed=1)
        rows.append(("rect", (hz,wz), wall, (0,0), r["found"], r["shape"], r["sign"], round(r["score"],3), round(r["coverage"],2), r["found"] and all(abs(a-b)<=1 for a,b in zip(r["shape"],(hz,wz))) and r["sign"]=="raised"))
    # --- 3. recessed square (hole): model as wall nearer with a far block = box with negative depth
    wall, depth = 300, -80; z = S.zone_mm(wall)
    fr = S.frames(wall, [S.Box(0,0,3*z*1.05,3*z*1.05,depth)], n=20, rng=5); r = run_scene(fr, seed=2)
    rows.append(("recessed", 3, wall, (0,0), r["found"], r["shape"], r["sign"], round(r["score"],3), round(r["coverage"],2), r["found"] and r["sign"]=="recessed"))
    # --- 4. controls: bare wall, tilted wall, shallow step (2 cm), tiny object (1 zone), two objects
    ctrl = [("wall", S.frames(300, [], n=20, rng=7)),
            ("tilted wall", S.frames(300, [], n=20, tilt_x=0.12, tilt_y=0.08, rng=8)),
            ("shallow 2cm step", S.frames(300, [S.Box(0,0,3*S.zone_mm(280)*1.05,3*S.zone_mm(280)*1.05,20)], n=20, rng=9)),
            ("1-zone object", S.frames(300, [S.Box(0,0,S.zone_mm(220)*0.9,S.zone_mm(220)*0.9,80)], n=20, rng=10)),
            ("two 2x2 blocks", S.frames(300, [S.Box(-2*S.zone_mm(220),0,2*S.zone_mm(220)*1.05,2*S.zone_mm(220)*1.05,80),
                                              S.Box( 2*S.zone_mm(220),0,2*S.zone_mm(220)*1.05,2*S.zone_mm(220)*1.05,80)], n=20, rng=11))]
    for name, fr in ctrl:
        r = run_scene(fr, seed=3)
        expect_none = name in ("wall", "tilted wall", "1-zone object")
        rows.append((name, "-", 300, "-", r["found"], r["shape"], r["sign"], round(r["score"],3), round(r["coverage"],2), (not r["found"]) if expect_none else None))
    # --- report
    print(f"{'scene':18s} {'target':8s} {'wall':>5s} {'offset':>12s} {'found':>5s} {'shape':>7s} {'sign':>9s} {'score':>6s} {'cov':>5s} {'ok':>4s}")
    for row in rows:
        name, tgt, wall, off, found, shape, sign, sc, cov, ok = row
        print(f"{name:18s} {str(tgt):8s} {wall:5d} {str(off):>12s} {str(found):>5s} {str(shape):>7s} {str(sign):>9s} {sc:6.3f} {cov:5.2f} {'' if ok is None else ('yes' if ok else 'NO'):>4s}")
    judged = [r for r in rows if r[-1] is not None]
    print(f"\n{sum(1 for r in judged if r[-1])}/{len(judged)} judged scenes correct (size within one zone, sign right), {time.time()-t0:.0f}s total")
    exact = [r for r in rows if r[0]=="square" and r[4] and r[5]==(r[1],r[1])]
    print(f"{len(exact)}/{sum(1 for r in rows if r[0]=='square')} squares read at exactly the geometric size")

if __name__ == "__main__":
    main()
