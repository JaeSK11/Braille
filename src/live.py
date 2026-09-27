#!/usr/bin/env python3
"""Sensor -> 5x5 field -> 25-qubit circuits -> Aer -> square detection, in a loop.

    python3 live.py                 # live from /dev/ttyACM0
    python3 live.py --npy scene.npy # from a saved (N,8,8) recording
"""
import argparse, sys, time
import numpy as np
import acquire, grid, run_sim, detect, identify

def one_pass(frames, a):
    h, d5, win, std5, (r0, c0) = grid.frames_to_h(frames, a.r0, a.c0, a.dmin, a.dmax, auto=a.auto, binary=a.binarise,
                                                   quality=identify.library_quality if a.identify else None)
    edge_p11, z_p1 = run_sim.run(h, shots=a.shots, noisy=not a.ideal)
    res = detect.detect(edge_p11, z_p1, a.threshold, squares_only=a.squares_only)
    print("\033[2J\033[H" if not a.npy else "")
    print(f"{'split at' if a.binarise else 'depth window'} {win[0]:.0f}..{win[1]:.0f} mm   crop offset (r0,c0)=({r0},{c0}){' auto' if a.auto else ''}   shots/circuit {a.shots}   frames averaged {len(frames)}")
    print("\n5x5 depth (mm):");  print(np.array2string(d5, precision=0, suppress_small=True, max_line_width=120))
    print("\nh (angle/pi):");    print(np.array2string(h, precision=2, max_line_width=120))
    print("\nedges (=== strong, --- weak):"); print(detect.edge_grid_str(edge_p11))
    hgt, wid = res["shape"]
    name = "SQUARE" if hgt == wid == 3 else f"RECTANGLE {hgt}x{wid}"
    verdict = f"{name} at {res['pos']} ({res['sign']})" if res["found"] else "no block"
    if res["found"]:
        hm, wm, zone = grid.size_estimate(d5, res["shape"])
        verdict += f"   ~{hm/10:.0f} x {wm/10:.0f} cm (+-{zone/10:.0f} cm)"
    print(f"\n{verdict}   score {res['score']:.3f}  coverage {res['coverage']:.2f}  boundary {res['boundary_p11']:.3f}  flat {res['flat_p11']:.3f}")
    if a.identify:
        mean8 = grid.orient(grid.average_frames(frames)[0])
        if grid.plateau(mean8, r0, c0):
            print("shape: filled_all_25 (raised plateau fills the window; seen from the 8x8 ring)")
        elif identify.all_flat(edge_p11):
            print("shape: flat (no edges in the 5x5)")
        else:
            s = identify.identify(edge_p11, z_p1)
            if s["found"]:
                import shapes as _sh
                print(f"shape: {s['name']} at {s['pos']} ({s['sign']})   score {s['score']:.3f}  coverage {s['coverage']:.2f}")
                print(_sh.show(s["mask"]))
            else:
                print(f"shape: none matched (best {s.get('name')}, score {s.get('score',0):.3f}, coverage {s.get('coverage',0):.2f})")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy"); ap.add_argument("--port")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--r0", type=int, default=1); ap.add_argument("--c0", type=int, default=1)
    ap.add_argument("--dmin", type=float); ap.add_argument("--dmax", type=float)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--ideal", action="store_true", help="no noise model")
    ap.add_argument("--auto", action="store_true", help="centre the 5x5 crop on the closest 3x3 block")
    ap.add_argument("--binarise", action="store_true", help="snap each cell to near/far (Otsu) before encoding")
    ap.add_argument("--squares-only", action="store_true", help="only test 3x3 templates")
    ap.add_argument("--identify", action="store_true", help="also match against the shape library (shapes.py)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.npy:
        one_pass(np.load(a.npy), a); return
    port = a.port or acquire.find_port()
    buf = []
    for g in acquire.frames(port):
        buf.append(g)
        if len(buf) >= a.frames:
            one_pass(np.array(buf), a); buf = []
            if a.once: break

if __name__ == "__main__":
    main()
