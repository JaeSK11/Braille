#!/usr/bin/env python3
"""Read the DFRobot SEN0628 8x8 ToF grid over USB serial.

Stream format (firmware V1.3, observed 2026-09-26):
    y0:4000,4000,4000,24,19,17,14,14,
    ...
    y7:...
One frame = rows y0..y7, values in mm, 4000 = no valid return.

Usage:
    python3 acquire.py                    # print frames live
    python3 acquire.py --frames 30 --save wall50cm.npy
    python3 acquire.py --port /dev/ttyACM1
"""
import argparse, glob, sys, time
import serial

INVALID = 4000

def find_port():
    ports = sorted(glob.glob("/dev/ttyACM*")) or sorted(glob.glob("/dev/ttyUSB*"))
    if not ports:
        sys.exit("no serial device found (expected /dev/ttyACM0). Is the sensor plugged in?")
    return ports[0]

def frames(port, baud=115200):
    """Yield 8x8 lists of ints, one per complete frame."""
    with serial.Serial(port, baud, timeout=2) as s:
        s.reset_input_buffer()
        grid = {}
        while True:
            line = s.readline().decode("ascii", "ignore").strip()
            if not line.startswith("y") or ":" not in line:
                continue
            tag, _, body = line.partition(":")
            try:
                r = int(tag[1:])
                vals = [int(v) for v in body.strip(",").split(",") if v]
            except ValueError:
                continue
            if len(vals) != 8:
                continue
            if r == 0:
                grid = {}
            grid[r] = vals
            if r == 7 and len(grid) == 8:
                yield [grid[i] for i in range(8)]
                grid = {}

def show(g):
    for row in g:
        print(" ".join("  ---" if v >= INVALID else f"{v:5d}" for v in row))
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    ap.add_argument("--save", default=None, help="save collected frames to .npy (needs numpy)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    port = a.port or find_port()
    print(f"reading {port} @115200", file=sys.stderr)
    got, t0 = [], time.time()
    for g in frames(port):
        got.append(g)
        if not a.quiet:
            show(g)
        if a.frames and len(got) >= a.frames:
            break
    dt = time.time() - t0
    print(f"{len(got)} frames in {dt:.1f}s = {len(got)/dt:.1f} fps", file=sys.stderr)
    if a.save:
        import numpy as np
        arr = np.array(got, dtype=np.int16)
        np.save(a.save, arr)
        valid = arr < INVALID
        print(f"saved {a.save} shape={arr.shape}", file=sys.stderr)
        print("per-zone mean (mm), invalid shown as ---:", file=sys.stderr)
        m = np.where(valid, arr, np.nan)
        for row in np.nanmean(m, axis=0):
            print(" ".join("  ---" if np.isnan(v) else f"{v:5.0f}" for v in row), file=sys.stderr)
        print("per-zone std (mm):", file=sys.stderr)
        for row in np.nanstd(m, axis=0):
            print(" ".join("  ---" if np.isnan(v) else f"{v:5.1f}" for v in row), file=sys.stderr)

if __name__ == "__main__":
    main()
