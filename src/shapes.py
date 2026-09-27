"""Shape library on the 5x5 grid, and how to realise each one physically in sensor_sim.

A shape is a 5x5 boolean mask of raised cells. `scene(mask, ...)` turns it into a wall
plus one Box per raised cell, so sensor_sim renders it like a real object made of blocks.
"""
import numpy as np
import sensor_sim as S

def M(rows):
    return np.array([[ch == "#" for ch in r] for r in rows], dtype=bool)

LIBRARY = {
    "filled_square_3x3": M([".....",
                            ".###.",
                            ".###.",
                            ".###.",
                            "....."]),
    "square_outline_3x3": M([".....",
                             ".###.",
                             ".#.#.",
                             ".###.",
                             "....."]),
    "square_outline_5x5": M(["#####",
                             "#...#",
                             "#...#",
                             "#...#",
                             "#####"]),
    "filled_square_4x4":  M([".....",
                             ".####",
                             ".####",
                             ".####",
                             ".####"]),
    "filled_all_25":      M(["#####",
                             "#####",
                             "#####",
                             "#####",
                             "#####"]),
    "plus":               M(["..#..",
                             "..#..",
                             "#####",
                             "..#..",
                             "..#.."]),
    "x_cross":            M(["#...#",
                             ".#.#.",
                             "..#..",
                             ".#.#.",
                             "#...#"]),
    "h_bar":              M([".....",
                             ".....",
                             "#####",
                             ".....",
                             "....."]),
    "v_bar":              M(["..#..",
                             "..#..",
                             "..#..",
                             "..#..",
                             "..#.."]),
    "diagonal":           M(["#....",
                             ".#...",
                             "..#..",
                             "...#.",
                             "....#"]),
    "L":                  M([".#...",
                             ".#...",
                             ".#...",
                             ".###.",
                             "....."]),
    "T":                  M([".###.",
                             "..#..",
                             "..#..",
                             "..#..",
                             "....."]),
    "U":                  M([".#.#.",
                             ".#.#.",
                             ".#.#.",
                             ".###.",
                             "....."]),
    "single_centre":      M([".....",
                             ".....",
                             "..#..",
                             ".....",
                             "....."]),
    "square_2x2":         M([".....",
                             ".##..",
                             ".##..",
                             ".....",
                             "....."]),
    "checkerboard":       M(["#.#.#",
                             ".#.#.",
                             "#.#.#",
                             ".#.#.",
                             "#.#.#"]),
    "two_dots":           M([".....",
                             ".#.#.",
                             ".....",
                             ".#.#.",
                             "....."]),
}

def show(mask):
    return "\n".join("".join("#" if v else "." for v in row) for row in mask)

def scene(mask, wall=300.0, depth=80.0, crop=(1, 1), jitter=0.0, rng=None):
    """Boxes for a 5x5 mask placed at 8x8 offset `crop` (row0, col0).
    Each raised cell becomes a Box exactly one zone wide at the face distance.
    `jitter` shifts the whole shape by that fraction of a zone (misalignment test)."""
    rng = np.random.default_rng(rng)
    z = S.zone_mm(wall - depth)
    r0, c0 = crop
    dx = dy = 0.0
    if jitter:
        dx, dy = rng.uniform(-jitter, jitter, 2) * z
    boxes = []
    for r in range(5):
        for c in range(5):
            if mask[r, c]:
                # zone centre in tangent-plane mm at the face distance; 8x8 centred on axis
                x = ((c0 + c) - 3.5) * z + dx
                y = ((r0 + r) - 3.5) * z + dy
                boxes.append(S.Box(x, y, z * 1.02, z * 1.02, depth))
    return wall, boxes

def frames(mask, n=20, wall=300.0, depth=80.0, crop=(1, 1), jitter=0.0, tilt_x=0.02, rng=None):
    w, boxes = scene(mask, wall, depth, crop, jitter, rng)
    return S.frames(w, boxes, n=n, tilt_x=tilt_x, rng=rng)

if __name__ == "__main__":
    for name, m in LIBRARY.items():
        print(f"{name}  ({int(m.sum())} cells)"); print(show(m)); print()
