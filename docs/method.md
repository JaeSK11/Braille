# Method

## 1. From frames to a 5x5 field (`src/grid.py`)

1. **Average** N frames per zone, ignoring `4000` (invalid). Zones with no
   valid sample are filled with the median of their valid neighbours.
2. **Orient** according to `orientation.json` (camera view by default).
3. **Crop** a 5x5 window from the 8x8. With `--auto`, the crop whose inner
   3x3 is nearest to the sensor is chosen.
4. **Detrend**: the wall is the far surface, so a plane is fitted to the
   farthest cells, the inlier set is grown, and the plane is subtracted. This
   is only applied when the fitted ramp exceeds half of `min_step` (15 mm).
5. **Snap** (`--binarise`): each cell is set to *near* (0.15) or *far* (0.85).
   Half-covered zones are ambiguous, so every candidate split between the two
   surface levels is tried (Otsu, midpoint, every gap in the sorted depths).
   The split kept is the one whose near-mask is most rectangular, or most like
   a library shape. Steps smaller than 30 mm are treated as flat.
6. Without snapping, depths are **normalised** into an automatic window
   around the scene and clipped to `h ∈ [0.1, 0.9]`. Normalising to the
   sensor's full range would shrink a step into the noise.

## 2. Encoding (`src/encode.py`)

```
θ_ij = π · h_ij
|q_ij⟩ = RY(θ_ij) |0⟩          qubit index = 5·row + col
```

This is one layer of single-qubit rotations: a product state of depth 1.

## 3. Readout

**Depth:** measure Z, `P(1) = sin²(θ/2)`.

**Edge strength between neighbours:** a Bell-basis measurement on the pair
(CNOT a→b, H on a, measure both):

```
P(11) = sin²((θ_a − θ_b)/2) / 2
```

A flat edge gives about 0 and a full step gives up to 0.5. A 5x5 grid has
**40 nearest-neighbour edges**, which are covered by four disjoint layers
(horizontal even/odd, vertical even/odd), so there are four Bell circuits.

**Sign:** the Bell readout cannot tell a raised block from a recessed one.
The Z circuit decides: the side with the lower mean `P(1)` is nearer, and
therefore raised.

**Inverse template (optional):** apply `RY(−π·t_ij)` for a template `t` and
count the ones. A match gives almost all zeros.

Budget per decision: 5 circuits (4 Bell + 1 Z) at 100 shots each.

## 4. Detection and identification

**Blocks (`src/detect.py`):** every rectangle from 2x2 to 4x4 at every
position has a known set of boundary edges. A 3x3 block, for example, touches
12 of the 40 edges. The score is the mean `P(11)` on the boundary edges minus
the mean on all other edges. The coverage is the fraction of boundary edges
above 0.06 minus the same fraction elsewhere. A block is reported when
score ≥ 0.15 and coverage ≥ 0.6.

**Shapes (`src/identify.py`):** the same scoring is run against the 17
library masks in `src/shapes.py` at every translation that fits in the grid,
plus a term that rewards masks splitting the Z readout into two tight groups.
A mask and its complement have identical edge sets, so Z decides which side
is raised.

Library: `filled_square_3x3`, `square_outline_3x3`, `square_outline_5x5`,
`filled_square_4x4`, `filled_all_25`, `plus`, `x_cross`, `h_bar`, `v_bar`,
`diagonal`, `L`, `T`, `U`, `single_centre`, `square_2x2`, `checkerboard`,
`two_dots`.

`filled_all_25` has no internal edges, so the qubits cannot see it. It is
detected classically beforehand (`grid.plateau`): the crop is uniformly nearer
than the surrounding 8x8 ring.

## 5. Simulation backend (`src/run_sim.py`)

Qiskit Aer with the **matrix product state** method (about 1–4 s per scene,
compared with more than 100 s for statevector at 25 qubits). The noise model
is a symmetric readout error (2%) on every qubit plus 2-qubit depolarising
error (1%) on every CNOT.

## 6. Hardware target

The product-state preparation has depth 1 and the Bell layers only touch
nearest neighbours. The circuits therefore map directly onto a square-lattice
superconducting chip, or onto a 25–36 qubit trapped-ion machine without
routing.
