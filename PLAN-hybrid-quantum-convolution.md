# QuantumLIDAR: hybrid quantum convolution

Plan written 2026-10-01. Companion to `PLAN-learned-shape-identifier.md`
(the model ladder this slots into) and `PLAN-5x5-quantum-depth-grid.md`
(the working rule-based pipeline). Nothing here is built yet.

---

## 1. The idea

Run convolution over the LiDAR data with a small variational quantum
circuit as the filter, and feed the resulting feature map to a classical
model that does the actual detection. The point-cloud form of this voxelizes
the local neighbourhood of each point, encodes the voxels into qubits, and
extracts features with the circuit. Fully quantum detection (state in, label
out, no classical head) is a much larger undertaking and is out of scope.

Short answer on feasibility: yes. Of the approaches in our literature notes,
hybrid quantum convolution is the only one with a published result on real
LiDAR (HQCOD / MC-QCNN on KITTI, section 10). It is also the smallest step
from what already works here.

## 2. We already have a fixed quantum convolution

The Bell-pair readout in `encode.py` is a quantum convolution with a fixed
kernel:

| Convolution part | Current pipeline |
|---|---|
| Patch | Two neighbouring cells (one edge) |
| Patch encoding | `RY(pi * h)` on each cell's qubit |
| Filter | Fixed circuit: `CX a->b, H a` |
| Feature | P(11), the edge gradient magnitude |
| Sliding | 40 edges, covered by 4 disjoint layers (`LAYERS`) |
| Classical head | `detect.py` template matching |

The proposal generalises this three ways:

1. **Bigger kernel**: 2x2 or 3x3 patches instead of pairs.
2. **Trainable kernel**: a variational circuit instead of `CX, H`.
3. **Learned head**: a small classical model instead of templates.

## 3. The constraint: the SEN0628 gives 64 points

Voxelized local neighbourhoods suit dense clouds (KITTI is ~100k points per
frame). The VL53L7CX gives an 8x8 grid, one perpendicular distance per zone.
Casting those 64 points into a 3D voxel grid:

- **Mostly empty voxels.** The scene is 2.5D: one surface per viewing
  direction. A 3D occupancy grid holds the same information as the range
  image, spread over many more qubits.
- **Lost depth precision.** RY encoding keeps depth as a continuous angle.
  Occupancy at 2 cm voxels reduces it to a few bits. Zones are already
  ~43 mm wide at 30 cm, so depth is the one axis where the sensor is
  precise (~5 mm noise), and voxelizing throws that away.

3D voxels become worth it when the cloud is genuinely 3D: frames fused from
several sensor poses, or a denser LiDAR. So the plan has two steps, and the
second is conditional.

## 4. Step 1: quantum convolution on the range image

Treat each zone as a voxel column whose height is the depth. This keeps
continuous depth and the grid's neighbourhood structure.

### 4.1 Input

- Full 8x8 frame (not the 5x5 crop): convolution handles position, so
  auto-crop is unnecessary and the plateau case (object filling the crop)
  sees its edges.
- Normalise with the existing window: `grid.normalise`, h in [0.1, 0.9],
  optionally after `grid.detrend`. Use the continuous field, not the
  binarised one, so the filter sees real depth.

### 4.2 Filter

| Choice | Kernel 2x2 | Kernel 3x3 |
|---|---|---|
| Qubits per patch | 4 | 9 |
| Patches on 8x8, stride 1 | 49 (7x7 map) | 36 (6x6 map) |
| Encoding | `RY(pi * h)` per cell, same as `encode.py` | same |
| Ansatz | L = 1 to 2 layers of [RY, RZ on every qubit] + CNOT ring | L = 1 to 2, CNOTs on patch nearest neighbours |
| Trainable angles | 8 to 16 per filter | 18 to 36 per filter |
| Readout | <Z> on each qubit -> 4 channels | <Z> on a subset (centre + 4 edge midpoints) -> 5 channels |

Start with 2x2. Use K = 1 to 4 filters (different random or trained
parameters), giving a 7x7x(4K) feature map.

Optional data re-uploading: repeat the RY encoding between ansatz layers.
Try with and without, as in rung 5 of the learned-identifier plan.

### 4.3 Classical head

- Tiny CNN: two 3x3 conv layers (16 channels), global average pool, linear
  layer to the 18 classes (17 library shapes + `none`), plus `other` once
  that generator exists.
- Alternative: flatten + MLP (2 x 64), to compare with rung 2.

### 4.4 Two training modes

1. **Fixed random filters (quanvolution).** Filter parameters drawn once at
   random and frozen. Features are computed once per scene and cached; only
   the classical head trains. This is the original quanvolution recipe and
   the cheapest version.
2. **End-to-end.** Filter angles and the head trained together. Gradients
   come from exact statevector simulation with backprop or adjoint
   differentiation, not parameter-shift with shots. At 4 to 9 qubits this
   is cheap.

Do mode 1 first. Mode 2 only if mode 1 shows the quantum features carry
something.

### 4.5 Cost

- A 4-qubit filter is a 16x16 unitary. With fixed filters the features can
  be computed in plain numpy: build the product state for every patch,
  apply the unitary, take <Z>. 20,000 scenes x 49 patches = ~1M patch
  evaluations, seconds to minutes on one core. PennyLane is not needed for
  mode 1.
- Mode 2 in PennyLane (`default.qubit` or `lightning.qubit`, torch
  interface, batched patches) or a hand-written torch 16x16 unitary.
  Minutes per epoch on CPU at this size.
- Compare with rung 5 (25 qubits on MPS, budgeted at one core-day).
  This is far cheaper.

### 4.6 Shot noise, to stay honest

Exact <Z> values are what a perfect, infinitely sampled device would give.
To compare with input B (real shot noise), add a mode that samples each
<Z> from a binomial at S shots (S = 50 to 500), and optionally the same
readout and CX noise as `run_sim.noise_model`. Report accuracy versus S.

### 4.7 Running it on hardware

Patches that do not overlap can share one circuit, the same trick the four
Bell layers use. On the 8x8 frame with a 2x2 kernel:

- Stride 2: 16 disjoint patches, 64 qubits, one circuit per filter.
- Stride 1: 4 offset classes ((0,0), (0,1), (1,0), (1,1)), each a set of
  disjoint patches, so 4 circuits per filter. The direct analogue of
  `H_even / H_odd / V_even / V_odd`.
- On the 5x5 crop: 25 qubits, which matches what is already simulated.

Shallow and nearest-neighbour only, so it maps onto a lattice chip.

## 5. Step 2 (conditional): 3D voxel quantum convolution

Do this only once there are multi-pose captures or a denser sensor.

### 5.1 Zones to points

The sensor reports perpendicular distance d per zone. With zone-centre
tangents `tx, ty` (from the 60 deg field, as in `sensor_sim.zone_rays`):

```
x = d * tx      y = d * ty      z = d
```

Multi-pose fusion also needs the sensor pose for each frame (a fixed rig
with known offsets, or ICP registration). Out of scope until the rig exists.

### 5.2 Local voxel neighbourhoods

- For each point, a cube centred on it, split into 2x2x2 voxels (8 qubits).
  Voxel size about one zone width at the working distance.
- Encoding options: occupancy as RY(0) or RY(pi); point count scaled to
  an angle; or the mean depth offset inside each voxel as the angle (keeps
  some of the precision occupancy loses).
- Same ansatz family as step 1, 8 qubits, <Z> per qubit -> 8 features per
  point. Pool features over points (max or mean) before the head, which
  also makes the result independent of point order.
- Head: PointNet-style (shared per-point MLP + max pool), or scatter the
  features back onto the 8x8 grid and reuse the step 1 CNN head.

### 5.3 Expected outcome on single SEN0628 frames

Roughly the same accuracy as step 1 at about twice the qubits per patch,
because of the 2.5D problem in section 3. Run it once on single frames to
confirm, so the reason for not pursuing it further is measured rather than
assumed.

## 6. Controls

Quantum convolution papers often match a classical convolution with random
nonlinear filters rather than beat it. Every quantum result here is reported
next to its control:

| Control | What it isolates |
|---|---|
| C1. No filter: raw 2x2 patches unfolded to a 7x7x4 map, same head | Does the filter add anything at all? |
| C2. Random classical filter: random 4->4 linear map + tanh (or cos), same output shape, same head | Quantum filter versus any random nonlinear filter |
| C3. Trained classical conv layer (rung 3 CNN) | Quantum filter versus a learned classical one |
| C4. Fixed Bell-edge features (input B) with the same head | New quantum filter versus the quantum feature we already have |

A quantum result counts if it beats C2 and matches or beats C3 with fewer
parameters, or degrades more gracefully than C3 as shots or frames drop.

## 7. Where it sits in the model ladder

Insert as rungs 4a and 4b in `PLAN-learned-shape-identifier.md` section 5,
between the classical rungs and the 25-qubit VQC:

| Rung | Model | Input | Params | Purpose |
|---|---|---|---|---|
| 4a | Fixed random quantum filters + CNN head | A, 8x8 | head ~5k, filters 0 trained | Cheapest quantum feature extractor |
| 4b | Trained quantum filters + CNN head | A, 8x8 | head ~5k + ~16 to 64 angles | Does training the filter help? |

Same data (`dataset.py`, synthetic from `sensor_sim` + `shapes`, real
recordings as Test 2), same splits, same evaluation curves (section 6 of
that plan). Lower risk than rung 5: small circuits, exact gradients, no
25-qubit barren plateau.

## 8. Implementation steps

1. `quanv.py`: patch extraction (8x8 -> patches), encoding, filter ansatz,
   exact <Z> features in numpy, optional shot sampling. Unit test: a flat
   field gives the same feature everywhere; a step edge changes features
   only on patches that straddle it.
2. `quanv_features.py`: compute and cache 4a features for the synthetic
   dataset, plus controls C1 and C2. Parallel over cores.
3. Extend `train_classical.py` with the CNN head on cached feature maps
   (rung 4a, C1, C2, C4).
4. `train_quanv.py`: rung 4b, end-to-end in PennyLane + torch.
5. Add 4a and 4b to `eval.py` curves; `live.py --model` works unchanged
   once the model wraps feature extraction.
6. Step 2 only if the conditions in section 5 are met.

Prerequisite: `dataset.py` from the learned-identifier plan, since both
plans train on the same scenes.

## 9. Risks and open questions

- **No quantum advantage.** The likely outcome is 4a about equal to C2. That
  is still a result, and the controls are designed to show it cleanly.
- **The task may be too easy.** If rung 1 already solves the aligned
  library, the interesting comparisons are the `other` class, misalignment,
  and accuracy versus shots.
- **Too little input.** 49 patches of 4 values is a small field; a 3x3
  kernel on 8x8 leaves only a 6x6 map. Padding (edge-replicate) is an
  option.
- **Open:** 2x2 or 3x3 first? 2x2 is cheaper and matches the Bell layers;
  3x3 sees a whole raised cell with its neighbours. Start 2x2.
- **Open:** detrended or raw input? Detrending removes wall tilt but bakes
  in a classical step the model might learn on its own. Try both.

## 10. Environment

Same as the learned-identifier plan: `~/venvs/qlidar` has numpy, qiskit,
qiskit-aer and pyserial. Mode 1 needs only numpy. Mode 2 needs
`pip install pennylane pennylane-lightning torch scikit-learn`.

## 11. Related literature

| Paper | What it does | Relevance | Status |
|---|---|---|---|
| Henderson et al., *Quanvolutional Neural Networks*, 2019, arXiv:1904.04767 | Fixed random quantum circuits as conv filters on images, classical CNN after | The recipe for mode 1 | `-`, verify citation |
| *Hybrid quantum-classical 3D object detection using multi-channel QCNN* (HQCOD / MC-QCNN), J. Supercomputing 2025, DOI 10.1007/s11227-025-06968-7 | Hybrid quantum conv for 3D detection, evaluated on KITTI | The only published real-LiDAR result; closest to step 2 | `-`, not downloaded |
| Heredge, Hill, Hollenberg, Sevior, *Permutation Invariant Encodings for QML with Point Cloud Data*, 2023, arXiv:2304.03601 | See below | Different approach; one idea to borrow | `pdf`, read |
| *HyQuRP*, 2026, arXiv:2602.06381 | Hybrid QNN with rotation and permutation equivariance for 3D point clouds | Follow-on to Heredge et al. | `-`, not downloaded |
| *Layered Quantum Architecture Search for 3D Point Cloud Classification*, 2026, arXiv:2603.20024 | Architecture search for point-cloud QNNs | May inform the filter ansatz | `pdf`, not read |
| *3D-QAE*, BMVC 2023, arXiv:2311.05604 | Fully quantum autoencoder for 3D point clouds | The "fully quantum" end of the spectrum | `pdf`, not read |

### Heredge et al. (2023) is not quantum convolution

Checked 2026-10-01, since it is in `papers/` and looks related.

- **Data:** synthetic clouds of 2 to 6 points sampled from a sphere or a
  torus. No LiDAR, no voxels, no neighbourhoods.
- **Encoding:** each point (x, y, z) gets its own 3-qubit register (best
  circuit: H then RZ(x), RZ(y), RZ(z)). The whole cloud is put into an
  equal superposition of every ordering of the points, so reordering the
  point list cannot change the state.
- **On hardware:** done probabilistically with controlled-SWAPs and
  O(n^2) ancillas, discarding runs where the ancillas are not |0>. In
  simulation they build the state directly at O(n!) classical cost.
- **Model:** quantum kernel SVM, not a variational circuit with a classical
  detection head.
- **Result:** sphere versus torus at ~95% with 3 points, beating PointNet
  and an RBF SVM on small datasets; plain IQP encoding gets worse as points
  are added. Noiseless statevector simulation only.

Why it does not fit the SEN0628: the sensor gives an ordered grid, so point
order is never ambiguous, and discarding it would throw away the
neighbourhood structure the edge detection uses. It also does not scale:
64 points would need ~190 data qubits and ~2,000 ancillas.

What to borrow: build the data's real symmetry into the model. For us that
is translation (handled by convolution) and the 8 rotations and reflections
of the square grid (augmentation first, an equivariant ansatz later, as the
learned-identifier plan already says). In step 2, pooling over points gives
order independence for free.
