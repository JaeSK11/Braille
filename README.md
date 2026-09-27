# Braille

Quantum compute processing of LiDAR depth data.

A small flash-LiDAR sensor produces an 8x8 range image. A 5x5 window of that
image is encoded onto **25 qubits**, one qubit per depth cell, and the
circuit's measurements are used to find edges and identify simple raised
shapes, the way a fingertip reads a Braille cell by its raised dots.

```
 SEN0628 ToF sensor ──USB serial──▶ 8x8 depth frames (mm)
                                        │  average, orient, crop 5x5,
                                        │  detrend wall, snap near/far
                                        ▼
                               5x5 field h ∈ [0.1, 0.9]
                                        │  RY(π·h) on each of 25 qubits
                                        ▼
            4 Bell-pair circuits (40 neighbour edges) + 1 Z circuit
                                        │  Qiskit Aer, MPS method, noise model
                                        ▼
         edge strengths P(11) + per-cell Z  ──▶  template match
                                        ▼
                  shape, position, raised / recessed
```

## Why this framing

A LiDAR range image is a **scalar field**: one depth per (row, column). The
direction of each ray is fixed by the sensor geometry and carries no scene
information. So the *value* (depth) goes on the qubit as a rotation angle, and
the *position* lives in the qubit index and its nearest-neighbour structure.
Gradients of the field (edges) are then read with two-qubit Bell measurements
between neighbours.

Quantum image encodings, quantum edge detectors and quantum point-cloud models
all exist (see [REFERENCES.md](REFERENCES.md)). None of the work we found runs
a sensor-native LiDAR range image through them.

**Being honest about the scope:** the encoded state is a product state, so a
classical computer does the same arithmetic instantly. This project does not
claim a quantum speed-up. It shows that the encode → measure → match pipeline
works end to end on real ToF sensor output and survives realistic gate and
readout noise. That pipeline is also the input layer for the variational
models planned in [docs/roadmap.md](docs/roadmap.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/acquire.py` | Read 8x8 frames from the sensor over USB serial; optionally save to `.npy` |
| `src/grid.py` | Frame averaging with hole fill, orientation, 5x5 crop, wall detrend, near/far snap, normalisation |
| `src/encode.py` | 25-qubit RY encoding, the four Bell edge layers, Z readout, inverse-template circuit |
| `src/run_sim.py` | Runs the circuits on Qiskit Aer (matrix-product-state method) with readout and CNOT noise |
| `src/detect.py` | Block/rectangle detection from the 40 edge values, raised/recessed sign from Z |
| `src/identify.py` | Matches edges against a 17-shape library at every in-grid translation |
| `src/shapes.py` | The 17-shape library and a helper that builds each shape as a physical scene |
| `src/sensor_sim.py` | Synthetic sensor: renders a wall + boxes scene into realistic 8x8 frames |
| `src/sweep.py` | 52 synthetic scenes through the full quantum pipeline |
| `src/shape_test.py` | Every library shape, aligned and misaligned, through the full pipeline |
| `src/live.py` | Live loop: sensor (or a saved recording) → pipeline → printed verdict |
| `sim/square5x5.py` | Standalone analytic noise simulation of the encoding (no SDK needed) |
| `docs/` | [Hardware](docs/hardware.md), [method](docs/method.md), [simulation](docs/simulation.md), [roadmap](docs/roadmap.md) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python3 sensor_sim.py     # print a synthetic 8x8 frame
python3 sweep.py          # 52 synthetic scenes through the 25-qubit pipeline (~20 s)
python3 shape_test.py     # 17 shapes x 2 alignments (~12 s)
python3 ../sim/square5x5.py
```

With the sensor attached (Linux, user in the `dialout` group):

```bash
python3 acquire.py                               # print live frames
python3 acquire.py --frames 30 --save scene.npy  # record
python3 live.py --auto --binarise --identify     # live detection
python3 live.py --npy scene.npy --auto --binarise --identify
```

## Results (synthetic)

All numbers come from the simulator in this repo and are reproducible with
the commands above. Noise model: 2% readout error per qubit, 1% depolarising
error per CNOT, 100 shots per circuit.

| Test | Result |
|---|---|
| Ideal encoding, raised 3x3 square | found at the correct position, score 0.31; flat and tilted planes ≤ 0.02 |
| `sweep.py`: 52 scenes (squares of 2–4 zones at 25/30/40 cm, sub-zone offsets, rectangles, a recess, controls) | 46 / 52 correct (size within one zone, sign right) |
| `shape_test.py`: 17 shapes aligned to zones | 17 / 17 |
| `shape_test.py`: 17 shapes at 0.3-zone misalignment | 15 / 17 with the fixed seeds (10 to 15 across other seeds) |

The pipeline was also run end to end on bench recordings from the real sensor
(bare wall, single box at several distances). The raised object was found with
the correct position and sign, and its size within one zone. **No recorded
sensor data is included in this repository.**

Known limits: half-covered zones are ambiguous (an object N.15 zones wide
reads as N or N+1). A raised area that fills the whole 5x5 window has no
internal edges, so it is detected classically from the surrounding 8x8 ring.

## Data policy

This repository contains code, a sensor simulator and documentation only. No
recordings from the sensor are included. `.gitignore` excludes `*.npy`,
`data/` and PDFs. Research papers are cited in [REFERENCES.md](REFERENCES.md),
not redistributed.

## License

GPL-3.0, see [LICENSE](LICENSE).
