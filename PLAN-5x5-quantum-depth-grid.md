# QuantumLIDAR: 5x5 depth grid on 25 qubits

Working plan written 2026-09-05, before the sensor arrived. Everything in
here was worked out in conversation; the numbers in section 6 come from a
noise simulation you can re-run with `sim/square5x5.py`.

---

## 1. Goal

Encode a small LIDAR depth grid onto a quantum circuit, one qubit per grid
cell, and detect a simple shape from the circuit's measurements.

First target: a 5x5 grid where the inner 3x3 block is raised and the outer
ring is recessed. Detect the square, its position, and whether it is raised
or recessed.

## 2. The framing that got us here

- A LIDAR range image is a **scalar field**: one number per (row, col).
  Gradients of that field give edges and surface normals.
- Raw LIDAR is scalars indexed by angle. The direction vectors are implied
  by the sensor geometry, so they carry no information about the scene.
- Therefore: put the **value** (depth) on the qubit, and let the **position**
  live in the qubit index and the chip's neighbour structure. Do not try to
  put x,y,z on one qubit; a qubit state only holds two real parameters.
- Prior art exists for every piece (quantum image encodings, quantum Sobel
  and Hadamard edge detection, quantum point-cloud encodings) but nobody has
  put a LIDAR range image through it. See `papers/` and section 10.

## 3. Hardware decision

**Chosen: DFRobot SEN0628, "Gravity: 8x8 Matrix ToF 3D Distance Sensor".**

| Item | Value |
|---|---|
| Ranging die | ST VL53L7CX (flash LiDAR: 940 nm VCSEL, SPAD array, direct ToF, per-zone histograms) |
| Onboard MCU | RP2040 running DFRobot firmware |
| Zones | 8x8 = 64 |
| Field of view | 60 deg x 60 deg, 90 deg diagonal, about 7.5 deg per zone |
| Range | 20 mm to 3.5 m |
| Accuracy (spec) | +-11 to 12 mm under 20 cm; +-5% (white) / +-6% (grey) beyond 20 cm |
| Rate | 15 to 60 Hz |
| Host interface | **USB-C direct to the computer**, appears as a serial port, 115200 baud, text output, 8x8 mode only |
| Also | Gravity I2C (addr 0x30..0x33, cascade up to 4) and UART; not needed |
| Firmware | UF2 drag-and-drop. Latest **V1.3 (2025-09-04)** fixes an invalid-value bug. Update on arrival. |
| Price | $22 at dfrobot.com, ~$24 AliExpress |
| In the box | sensor, Gravity cable, aluminium brackets, screws. **No USB-C cable.** |

Why this one: it is a genuine time-of-flight, single-photon LiDAR device,
gives a native 2-D grid, and needs no ESP32, Raspberry Pi, or soldering.
The RP2040 is the microcontroller we would otherwise have had to add.

Alternatives considered and why not (for now):

| Device | Type | Verdict |
|---|---|---|
| LDROBOT D500 / LD19 | 2-D spinning dToF LiDAR, 12 m | Real LiDAR stream, but one plane only; needs a tilt servo to get 5 rows |
| Bare VL53L8CX on AliExpress carrier | 8x8 dToF | Needs an STM32 Nucleo (84 KB firmware blob, small Arduinos cannot); board's TXS0104E shifter breaks USB-I2C bridges and ESP32-S3 |
| Pololu VL53L8CX carrier | 8x8 dToF | Known-good fallback, still needs a Nucleo |
| Orbbec Astra Pro | Structured light, 640x480 | Not LiDAR; legacy OpenNI2-only driver; min range 0.6 m. Good dense range image if ever wanted |
| Xbox Kinect v2 | Indirect (CW) ToF, 512x424 | Best cheap dense ToF for later classical gradient work; libfreenect2 |

## 4. Bring-up checklist (when it arrives)

1. Buy/find a **USB-C data cable** (not charge-only).
2. Firmware: hold BOOT, plug in, a USB drive appears; copy the V1.3 `.uf2`
   from the DFRobot wiki onto it. Board reboots.
3. Serial permissions on chugjug (once, then log out/in):
   ```
   sudo usermod -aG dialout counterbitio
   ```
4. Plug in normally. Expect `/dev/ttyACM0` (cdc_acm driver is present).
   ```
   ls -l /dev/ttyACM*
   ```
5. Raw look at the stream:
   ```
   python3 -m serial.tools.miniterm /dev/ttyACM0 115200
   ```
   (or `screen /dev/ttyACM0 115200`). Note the exact line format; the wiki
   only shows a Windows serial-monitor screenshot.
6. Python environment (nothing is installed on chugjug yet):
   ```
   python3 -m venv ~/venvs/qlidar && source ~/venvs/qlidar/bin/activate
   pip install pyserial numpy qiskit qiskit-aer matplotlib
   ```
7. Acquisition script: read frames, parse 64 ints (mm), reshape 8x8, print.
   Check orientation by waving a hand across the field: decide which array
   axis is "left-right" and whether rows need flipping.
8. Log ~30 frames of a flat wall at 50 cm. Per-zone mean and std give the
   real noise floor, which matters more than the spec sheet.

Environment facts already checked: chugjug is a desktop (X870E board),
32 cores, 123 GB RAM, free USB 2/3 ports, `cdc_acm` and `usb_storage`
present, user not yet in `dialout`, no arduino-cli/qiskit installed.

## 4a. Arrival notes (2026-09-26)

Tested first on a Windows machine (COM4), then moved to chugjug.

- Firmware updated to V1.3 by UF2 drop. The board does not report its
  version over USB; V1.3 behaviour confirmed by invalid zones reading
  exactly **4000** rather than holding a stale value.
- **Stream format**, one line per row, values in mm, trailing comma:
  ```
  y0:4000,4000,4000,24,19,17,14,14,
  ...
  y7:4000,4000,4000,21,16,13,11,15,
  ```
- Measured rate about **16 fps**.
- First frames showed the right five columns stuck at 9 to 24 mm, below
  the 20 mm minimum: something was almost touching the lens on that side.
  Check for the shipping film over the lens, the bracket or cable in the
  field, or the board lying face-down.
- On chugjug the sensor failed to enumerate on USB port `1-1` (kernel
  errors `-71`, "Device not responding to setup address") with the same
  cable that works on Windows. That port is the problem; use a rear port.
- `acquire.py` at the project root parses this format, prints live frames,
  and with `--frames N --save file.npy` logs frames and prints per-zone
  mean and std. User added to `dialout` on 2026-09-26; needs re-login.
- ModemManager is active on chugjug; it may hold a new ttyACM port for a
  few seconds after plug-in. Wait, or `sudo apt remove modemmanager`.

## 4b. Pipeline status (2026-09-26, sensor live on chugjug)

- Sensor enumerates on a **rear** USB port as `/dev/ttyACM0` (Raspberry Pi
  Pico USB ID 2e8a:000a). Front-panel port 1-1 fails; do not use it.
- Noise floor from `scene_desk.npy` (30 frames, cluttered desk, not a wall):
  zones seeing one stable surface at 45 to 52 cm have **std 4 to 6 mm**.
  Zones straddling two surfaces flicker (std 50 to 170 mm); that is the
  multi-target behaviour and will mark the beacon's boundary ring.
- Manual (`Sensor/Product Manual ... .pdf`, 33 pages): stream rows y0..y7
  are the Y axis **top to bottom**, values within a row are X **left to
  right**, in mm. Whether "left" is the sensor's left or the viewer's left
  still needs the hand-wave test; record the answer in `orientation.json`.
- Venv: `~/venvs/qlidar` with pyserial, numpy, qiskit 2.5, qiskit-aer 0.17,
  pypdf. Run scripts with `~/venvs/qlidar/bin/python`. `sudo` needed for the
  port until the next login (dialout added 2026-09-26).
- Scripts at project root, all working:

  | Script | Does |
  |---|---|
  | `acquire.py` | serial -> 8x8 frames; `--frames N --save f.npy` logs and prints mean/std |
  | `grid.py` | frame averaging with hole fill, orientation, 5x5 crop, auto depth window, h in [0.1,0.9] |
  | `encode.py` | 25-qubit RY encoding, 4 Bell layers, Z readout, inverse template |
  | `run_sim.py` | Aer **matrix_product_state** method with readout + CX noise; returns 40 edge P(11) and 25 Z P(1). ~1 to 4 s per scene (statevector method took 106 s) |
  | `detect.py` | 9 square templates, score, raised/recessed sign from Z |
  | `live.py` | sensor or `--npy` recording -> full pipeline -> printout, looping |

- Qiskit results on synthetic 5x5 scenes match the standalone simulation:
  raised/recessed/top-left squares found at the right position with score
  ~0.31; flat and tilted planes score < 0.01. Desk recording: no square,
  score 0.12 (cluttered scene, below the 0.15 threshold).

- **Wall check passed (2026-09-27, `wall50.npy`)**: all 64 zones valid over
  30 frames, means 318 to 353 mm (wall was at ~33 cm), std 3.3 to 9.3 mm,
  typically 5 mm. A gentle ramp of ~35 mm from bottom-left to top-right is
  sensor tilt relative to the wall, harmless for the beacon.

- **Orientation verified (2026-09-27)**: the stream is a straight camera
  view. Row y0 is the top; column x0 is the sensor's left, which is the
  right of a person facing the lens. `orientation.json` written with no
  flips. When standing behind the sensor looking at the beacon, the printed
  grid matches what you see.

Next: build the beacon, then
`sudo ~/venvs/qlidar/bin/python live.py`.

## 4c. First beacon runs (2026-09-27)

Object used: a small box, 6.9 x 8.1 cm face, about 8.5 cm deep, against
the wall. Recordings `beacon1..7.npy` at wall distances from 16 to 42 cm.

What was learned:

- **Zone size is set by the distance to the object face, not the wall.**
  Zone width = 0.144 x distance. The box face is 8.5 cm nearer than the
  wall, so it subtends more zones than a wall-distance estimate says.
- **Zones that straddle the box edge return intermediate depths** and
  flicker between the two targets (std 50 to 170 mm, dropped frames).
  A 3.7-zone box therefore shows as a 2x2 solid core plus a ring of
  partial zones. Fix: Otsu two-class snap of the 5x5 depths to near/far
  before encoding (`--binarise`). Partial zones then fall on one side.
- **Auto-centring** (`--auto`) picks the 5x5 crop whose inner 3x3 is
  nearest, so the box only has to be roughly centred in the 8x8.
- **Sign fixed**: h is normalised depth, so smaller h = closer = raised.
- **Coverage check** added: the fraction of a template's boundary edges
  above P(11)=0.06 minus the fraction elsewhere; a block must reach 0.6.
- **Rectangle templates** added (2x2 .. 4x4). `--squares-only` restricts
  to 3x3.

Results with `--auto --binarise`:

| Recording | Wall | Verdict | Coverage |
|---|---|---|---|
| wall50 | 33 cm | no block | 0.00 |
| beacon4 | 38 cm | raised 2x2 at (1,2) | 0.88 |
| beacon6 | 21.5 cm | raised 4x4 at (1,1) | 0.62 |
| beacon7 | 16 cm | box fills the crop, wall only on one side | - |
| beacon8 | 27.5 cm | **raised 2x2 at (1,1), score 0.39, coverage 1.00, flat 0.012** | 1.00 |

beacon8 is the reference result: every boundary edge strong, every other edge flat, numbers matching the noise simulation. beacon4 was the first clean end-to-end result: real LiDAR frames -> 25
qubits -> Bell-pair readout -> correct shape, position and sign.

For this box to read as **3x3**, the wall should be at about **29 cm**
(wall zones read ~290 mm). A square object of 9 cm or more on a side at
20 to 25 cm is the easier long-term beacon.

Standard live command now:
```
sudo ~/venvs/qlidar/bin/python live.py --auto --binarise
```

## 4d. Sensor simulator and synthetic sweep (2026-09-27)

`sensor_sim.py` renders a scene (wall distance, tilt, list of `Box(x, y, w, h,
depth)` in mm) into (N,8,8) frames with the measured behaviours: 60 deg
square field, perpendicular distance, 5 mm noise (8 mm under 200 mm),
partial-zone smearing/flicker/dropouts, 4000 sentinel, 20 mm minimum.
Running it directly prints a synthetic beacon8 next to the real one.

`sweep.py` pushes 52 synthetic scenes through the full quantum pipeline
(grid -> 25 qubits -> Aer MPS -> detect): squares of 2, 3, 4 zones at 25,
30, 40 cm and five sub-zone offsets; three rectangles; a recess; controls
(bare wall, tilted wall, 2 cm step, 1-zone object, two blocks). 18 s total.

Snapping (`grid.binarise`) now tries every candidate split between the two
surface levels and keeps the one whose object mask is most rectangular,
ties to the larger mask, then the larger gap. Reason: a zone the object
covers by ~50% is genuinely ambiguous; Otsu and the midpoint each got some
real recordings wrong.

Results:

| Set | Result |
|---|---|
| Synthetic, size within 1 zone and sign right | 48 / 52 |
| Synthetic, size exactly the geometric size | 19 / 45 squares |
| Synthetic misses | all four at a half-zone vertical offset; fail safe as "no block" (coverage 0.47..0.56) |
| Real: wall50 | no block |
| Real: beacon4 (38 cm) | raised 2x2, ~10x10 cm +-5 |
| Real: beacon6 (21.5 cm) | raised 4x4, ~8x8 cm +-2, coverage 1.00 |
| Real: beacon8 (27.5 cm) | raised 2x2, ~7x7 cm +-3, coverage 1.00 |
| Box truth | 6.9 x 8.1 cm |

Quantisation is the physical limit: an object N.15 zones wide reads as N
or N+1 depending on where its edges fall relative to zone boundaries. The
size estimate printed by `live.py` carries +-1 zone for that reason.

Controls: wall, tilted wall and a 1-zone object correctly give nothing.
A 2 cm step is rejected on purpose (`min_step` 30 mm in `binarise`).

## 4e. Shape library and identification (2026-09-27)

`shapes.py` holds 17 named 5x5 masks and turns any mask into a physical
scene for `sensor_sim` (one block per raised cell, one zone wide, at a
chosen wall distance and stand-off, with optional sub-zone jitter).
`identify.py` matches the 40 measured edge strengths against the edge set
of every library shape at every in-grid translation; the Z readout picks
which side of the boundary is raised, so a mask and its complement are
told apart. `shape_test.py` renders every shape aligned and with a 0.3-zone
misalignment and identifies it through the full 25-qubit pipeline.
`live.py --identify` prints the matched shape and its mask.

Library: filled_square_3x3, square_outline_3x3, square_outline_5x5,
filled_square_4x4, filled_all_25, plus, x_cross, h_bar, v_bar, diagonal,
L, T, U, single_centre, square_2x2, checkerboard, two_dots.

| Test | Result |
|---|---|
| All 17 shapes, aligned to zones | 17 / 17 |
| All 17 shapes, 0.3-zone misalignment | 15 / 17 (single cell vs 2x2 swap; coin-flip partials) |
| Rectangle sweep (52 scenes) | 46 / 52 within one zone |
| Real recordings | wall: flat; beacon4: square_2x2; beacon6: filled_square_4x4; beacon8: square_2x2 |

Limits found and handled:

- **filled_all_25** has no edges inside the 5x5, so the 25 qubits alone
  cannot see it. `grid.plateau` spots it from the 8x8 ring (crop uniformly
  nearer than the ring) before the quantum step.
- **Wall tilt**: beacon4 had a 40 mm ramp across the crop from an
  off-square sensor, as large as the partial-zone steps. `grid.detrend`
  fits a plane to the farthest cells and flattens the wall, but only when
  the ramp exceeds 15 mm; a level wall is left alone.
- **Half-covered zones** remain the one ambiguity. Snapping picks the
  threshold whose mask is most rectangular or most like a library shape,
  ties to the larger mask, then the larger depth gap (`grid.binarise`).
- **Single cell** needed the near level taken from the minimum, not the
  10th percentile, and 1-cell masks allowed.

Physical recipe for a shape test: blocks of one zone width at the face
distance (zone = 0.144 x distance). At 30 cm wall / 8 cm stand-off a zone
is 3.2 cm, so 3 cm cubes (dice, sugar cubes, LEGO 4x4 bricks) placed on a
5x5 grid of 3.2 cm pitch reproduce any library shape.

## 5. Beacon build

| Parameter | Value | Reason |
|---|---|---|
| Standoff | 50 cm | Zones are ~6.5 cm wide here; 5x5 crop covers ~33 cm |
| Raised block | 20 cm x 20 cm, centred | Fills the inner 3x3 zones |
| Step height | 10 to 15 cm | Keeps the step at >= 30% of a 0.5 m depth window |
| Surfaces | Matte, light grey or white, both levels | Spec accuracy is for white/grey; avoid gloss |
| Background | Flat board >= 40 cm square | Fills the outer ring cleanly |
| Mount | DFRobot bracket, fixed and level | Repeatability between runs |

Crop the 5x5 from the middle of the 8x8 (rows 1..5, cols 1..5 or 2..6;
pick the one whose centre zone lands on the block centre).

Depth window: choose `d_min`, `d_max` around the scene (e.g. 35 cm to
85 cm), **not** the sensor's full range. Normalise
`h = (d - d_min) / (d_max - d_min)`, clip to [0.1, 0.9].

## 6. Quantum encoding and readout

### Encoding (one layer, product state)
```
theta_ij = pi * h_ij
qubit_ij = RY(theta_ij) |0>          # 25 qubits, index = 5*row + col
```
Optional: RZ with a second per-cell value (intensity/confidence) later.

### Readouts
- **Depth back out**: measure Z. `P(1) = sin^2(theta/2)`.
- **Gradient magnitude on each edge**: Bell measurement on the pair
  (CX a->b, H a, measure both). `P(11) = (1 - F)/2`,
  `F = cos^2((theta_a - theta_b)/2)`. Flat edge -> ~0, big step -> up to 0.5.
  A 5x5 grid has 40 edges; cover them with 4 circuits (horizontal even,
  horizontal odd, vertical even, vertical odd).
- **Gradient sign / raised vs recessed**: Bell readout is sign-blind. Use
  either Z reads of one inner + one outer cell, or the inverse-template
  circuit below.
- **Inverse template**: apply `RY(-pi * t_ij)` for a template `t`, measure
  all, count ones. A match gives ~0.5 ones across 25 qubits (readout noise),
  a mismatch gives many. One circuit per template; sign-sensitive.

### Square detection
A 3x3 block touches exactly **12 of the 40 edges**. Score for a candidate
position = mean P(11) on its 12 boundary edges minus mean P(11) on the other
28. Evaluate all 9 positions from the same 4 circuits; take the argmax; call
it a square if the score clears a threshold (~0.15 with the numbers below).

### Simulation results (`sim/square5x5.py`, 2% readout error, 1% CNOT depol, 100 shots/circuit)

| Scene | Best pos | Score | Inverse-template ones |
|---|---|---|---|
| Raised square, centre | (1,1) correct | 0.31 | 0.5 |
| Recessed square, centre | (1,1) correct | 0.32 | 16.4 |
| Raised square, top-left | (0,0) correct | 0.28 | 6.8 |
| Square + 5% depth jitter | (1,1) correct | 0.31 | 0.6 |
| Raised plus shape | none | 0.11 | 5.6 |
| Single raised pixel | none | 0.11 | 5.3 |
| Tilted plane | none | 0.01 | 7.1 |
| Flat plane | none | 0.01 | 5.4 |

Step size vs detectability (noise floor on a flat edge ~0.012):

| Step as fraction of depth window | Boundary P(11) |
|---|---|
| 0.10 | 0.012 (buried) |
| 0.16 | 0.03 (marginal) |
| 0.30 | 0.10 (clear) |
| 0.60 | 0.33 |

Budget for one decision: 4 Bell circuits + 1 Z or template circuit,
100 shots each.

### Circuit sketch (Qiskit)
```python
from qiskit import QuantumCircuit
import math

def encode(h):                        # h: 5x5 floats in [0,1]
    qc = QuantumCircuit(25, 25)
    for r in range(5):
        for c in range(5):
            qc.ry(math.pi * h[r][c], 5*r + c)
    return qc

def bell_layer(qc, pairs):
    for a, b in pairs:
        qc.cx(a, b); qc.h(a)
        qc.measure([a, b], [a, b])
    return qc

H_even = [(5*r+c, 5*r+c+1) for r in range(5) for c in (0, 2)]
H_odd  = [(5*r+c, 5*r+c+1) for r in range(5) for c in (1, 3)]
V_even = [(5*r+c, 5*(r+1)+c) for c in range(5) for r in (0, 2)]
V_odd  = [(5*r+c, 5*(r+1)+c) for c in range(5) for r in (1, 3)]
```

## 7. Software plan (in order)

1. `acquire.py`: serial -> 8x8 int array, frame averaging, CSV/NPY logging.
2. `grid.py`: crop 5x5, normalise to depth window, orientation fix.
3. `encode.py`: build the 25-qubit circuits above from a 5x5 array.
4. `run_sim.py`: Aer simulator with a noise model; produce the 40 edge values.
5. `detect.py`: 9-template square score, sign via Z or inverse template.
6. Loop: live sensor -> detection printout. Move the block; watch the
   position follow.
7. Hardware run: a 25 to 36 qubit trapped-ion backend (all-to-all, no
   routing) or a square-lattice superconducting chip. Product state prep is
   depth 1, so this is well within reach.

Honest framing: the state is a product state throughout, so a laptop does
the same arithmetic instantly. The result is that the encode-measure-match
pipeline survives real noise on real LiDAR data, which nobody has published.

## 8. Ideas queued after the square works

- **Confidence as mixedness**: per-zone sigma from the VL53L7CX as a partial
  depolarisation; Bell fidelity then auto-downweights noisy edges. Needs
  custom RP2040 firmware to expose sigma (DFRobot's stream is distances only;
  the board is open and ST's ULD runs on RP2040).
- **Multi-target zones**: the chip reports two distances in a zone that
  straddles the block edge. Also hidden behind the firmware for now.
- **Amplitude encoding comparison**: same 5x5 (or 8x8) in 5 or 6 qubits with
  QHED-style Hadamard gradient. Compare loading depth, shots, noise against
  the 25-qubit version.
- **MPS loading of smooth range images**: bond dimension needed per row as
  an edge detector in itself.
- **Scale**: 8x8 = 64 qubits (simulator fine, some hardware); 2x2 cascade of
  SEN0628 = 16x16.
- **Dense fields**: add a Kinect v2 (libfreenect2) when thousands of cells
  are wanted for classical gradient/normal work alongside.
- **Combinatorial track**: qubit-per-point as an Ising/MIS problem; neutral
  atom layouts that mirror point positions; prior art from Golyanik's group
  (Q-Match, 3D-QAE).

## 9. Ways to get this wrong (checklist)

- Charge-only USB-C cable: no serial port appears.
- Not in `dialout`: permission denied on `/dev/ttyACM0`.
- Old firmware (< V1.3): stray invalid values in the matrix.
- Depth window set to sensor full range: step becomes a few percent and
  disappears into the noise floor.
- Glossy or black beacon surfaces: accuracy spec does not hold.
- Row/col orientation flipped: template positions come out mirrored.
- Forgetting that Bell readout is sign-blind: raised and recessed look the
  same until you read Z.

## 10. Literature

Collected PDFs and citations, with an open-gaps list:
`papers/quantum-encoding-point-clouds-and-fields/README.md`

Most relevant to this plan: Yao et al. 2017 (QPIE + Hadamard edge
detection), the 2026 quantum Sobel gradient paper (arXiv 2605.00744),
the NEQR-vs-FRQI comparison (GLSVLSI 2025), and Li et al. 2024 on exact
symmetries for point-cloud QNNs. The one published result on real LiDAR
points (hybrid QCNN on KITTI, J. Supercomputing 2025) is paywalled.

## 11. Links

- SEN0628 product: https://www.dfrobot.com/product-2999.html
- SEN0628 wiki (firmware, USB output, API): https://wiki.dfrobot.com/sen0628/
- Getting started / firmware V1.3: https://wiki.dfrobot.com/sen0628/docs/21564
- API and protocol: https://wiki.dfrobot.com/sen0628/docs/21555
- VL53L7CX datasheet: search "VL53L7CX datasheet" on st.com
