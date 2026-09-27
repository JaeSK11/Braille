# Roadmap

## Next: a learned shape identifier

The template identifier is exact for library shapes that are aligned to
zones, and it is fully inspectable. It cannot handle shapes outside the
library, objects that are not blocks, half-zone misalignment, or several
objects at once. The next step asks a measurable question:

> Does putting the depth field through qubits give a model anything that a
> classical model on the same depths does not?

### Inputs compared

| Input | Size | Notes |
|---|---|---|
| A. Raw depth field | 25 | Classical baseline, no quantum step |
| B. Quantum readout | 40 edge P(11) + 25 Z = 65 | What the circuits return, including shot and gate noise |
| C. Quantum state | 25 qubits | `encode(h)` becomes the input layer of a variational circuit |

### Model ladder

Each rung must beat the rung below it on the real-sensor test set to justify
its extra complexity.

| Rung | Model | Input |
|---|---|---|
| 0 | Template identifier (current) | B |
| 1 | Logistic regression | A |
| 2 | MLP (2 x 64) | A, then B |
| 3 | Tiny CNN | A |
| 4 | MLP on B with a noise curriculum | B |
| 5 | Variational quantum classifier: `encode(h)`, then L x [RY, RZ, nearest-neighbour CZ], optional data re-uploading, Z on 5 qubits, linear map to classes | C |
| 6 | Hybrid: rung 5 plus a small classical head | C |

### Data

- **Training:** synthetic scenes from `sensor_sim` and `shapes`, sampled
  over wall distance (200–450 mm), stand-off (40–120 mm), jitter (±0.5
  zone), wall tilt, crop offset, frames averaged, and shots per circuit.
  Target: 20,000 scenes. An `other` class of random blobs, off-library
  rectangles and pairs of blocks.
- **Test 1:** held-out synthetic data outside the training ranges.
- **Test 2:** real sensor recordings of cube-built shapes and everyday
  objects. Kept private and not published.

### Evaluation

Accuracy and confusion matrices. Accuracy plotted against jitter, shots per
circuit and frames averaged. Parameter count and circuits per decision.

## Further ideas

- **Confidence as mixedness:** use per-zone range sigma as partial
  depolarisation, so noisy edges are down-weighted automatically. This needs
  custom RP2040 firmware to expose sigma.
- **Multi-target zones:** the VL53L7CX can report two targets per zone. This
  is currently hidden by the stock firmware.
- **Amplitude encoding comparison:** QPIE and Hadamard edge detection on the
  same field with 5–6 qubits, compared with the 25-qubit version.
- **Scale:** the full 8x8 grid on 64 qubits, and a 2x2 cascade of sensors for
  16x16.
- **Hardware runs:** a square-lattice superconducting chip or a trapped-ion
  backend.
