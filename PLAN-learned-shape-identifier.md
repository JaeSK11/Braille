# QuantumLIDAR: learned shape identifier

Plan written 2026-09-27. Companion to `PLAN-5x5-quantum-depth-grid.md`,
which describes the working rule-based pipeline this builds on.

---

## 1. Why a learned version

The template identifier is exact for the seventeen library shapes when
they are aligned to zones, and it is fully inspectable. It cannot handle
what it has not been told about: shapes off the list, objects that are not
clean blocks (a mug, a hand, a bottle), half-zone misalignment, and two
objects at once. A learned model trades inspectability for generalisation.

The design question that makes this more than a classifier exercise:
**does putting the depth field through qubits give the model anything a
classical model on the same depths does not?** The plan is built so that
question gets a measured answer rather than an assumed one.

## 2. What the model sees

Three candidate input representations, all already produced by the
pipeline. The experiment compares them.

| Input | Size | Source | Notes |
|---|---|---|---|
| A. Raw depth field | 25 (or 64 for 8x8) | `grid.frames_to_h`, no snapping | Classical baseline. No quantum step. |
| B. Quantum readout | 40 edge P(11) + 25 Z P(1) = 65 | `run_sim.run` | What the circuits actually return, with shot noise and gate noise baked in. |
| C. Quantum state, end to end | 25 qubits | `encode.encode` as the first layer of a variational circuit | The encoding layer becomes the input layer of a QNN. |

A and B are features for a classical model. C is a quantum model.
B is the interesting middle: a classical model that learns to read noisy
quantum measurements.

## 3. Labels

One record per scene:

| Field | Values |
|---|---|
| `shape` | one of the 17 library names, `none`, or `other` (see 4.3) |
| `pos` | (row, col) of the mask's top-left in the 5x5, or null |
| `sign` | raised / recessed / null |
| `size_zones` | (rows, cols) bounding box in zones |
| `jitter` | true sub-zone offset (dx, dy) in zones, for regression later |

First model predicts `shape` only. Position, sign and jitter are follow-on
heads once shape works.

## 4. Data

### 4.1 Synthetic, from `sensor_sim` + `shapes`

The simulator is validated against five real recordings, so it is the bulk
source. Per scene, sample:

| Parameter | Range |
|---|---|
| shape | uniform over 17 library shapes + `none` (bare wall) |
| wall distance | 200 to 450 mm |
| stand-off (depth) | 40 to 120 mm |
| jitter | uniform in [-0.5, 0.5] zones on each axis |
| wall tilt | 0 to 0.15 mm/mm on each axis |
| crop offset | random within the 8x8, model sees auto-crop like live does |
| frames averaged | 5 to 30 (fewer frames = noisier) |
| shots per circuit | 50 to 200 (for input B) |

Target: 20,000 scenes. Cost: input A is instant; input B at ~1 to 4 s per
scene on the MPS simulator is 6 to 20 hours on one core, so run it across
the 32 cores here (about 40 minutes). Cache B features to disk once.

### 4.2 Real, from the sensor

Recordings are the test of record, not training data at first.

- Existing: `wall50`, `scene_desk`, `beacon4`, `beacon6`, `beacon8`.
- To collect: each library shape built from 3 cm cubes on a 3.2 cm grid at
  30 cm, at three positions and two small misalignments, 30 frames each.
  17 shapes x 6 = ~100 recordings, about an afternoon.
- Later: non-block objects (mug, bottle, hand, book on edge) for the
  `other` class and for the generalisation test.

Store as `data/real/<shape>_<pos>_<n>.npy` with a `labels.csv`.

### 4.3 The `other` class

Objects that are raised but match no template. Synthetic: random blobs
(union of 2 to 6 random cells, connected), random rectangles outside the
library sizes, two blocks at once. Real: the non-block objects above.
This is what the template method cannot do and the learned one should.

### 4.4 Splits

- Train / validation: synthetic, 90 / 10, split by scene seed.
- Test 1: held-out synthetic at parameter values outside the training
  ranges (wall 450 to 550 mm, jitter at exactly 0.5).
- Test 2: all real recordings. This is the number that matters.

## 5. Models, in a ladder

Each rung must beat the one below on Test 2 to justify its complexity.

| Rung | Model | Input | Params | Purpose |
|---|---|---|---|---|
| 0 | Template identifier | B | 0 | Current baseline. Known: 17/17 aligned, 15/17 at 0.3 jitter. |
| 1 | Logistic regression | A | ~500 | Is the problem even hard? |
| 2 | Small MLP, 2 hidden layers of 64 | A, then B | ~10k | Classical ceiling on each input. |
| 3 | Tiny CNN on the 5x5 (or 8x8) | A | ~5k | Translation handled by architecture, not data. |
| 4 | MLP on B with noise curriculum | B | ~10k | Does a model learn to read noisy quantum features better than the templates? |
| 5 | Variational quantum classifier | C | ~100 to 300 rotation angles | The quantum model. |
| 6 | Hybrid: rung 5 outputs -> small classical head | C | rung 5 + ~1k | Usual practical form. |

### Rung 5 in detail

- Input layer: the existing `encode(h)`, 25 RY rotations. Unchanged.
- Trainable layers: L repetitions of [RY, RZ on every qubit] followed by
  CZ on nearest-neighbour pairs on the 5x5 grid (same four edge layers we
  already use). L = 2 to 4. Nearest-neighbour entanglers keep MPS
  simulation tractable and match a square-lattice chip.
- Data re-uploading: optionally repeat `encode(h)` between layers. This is
  how the point-cloud QNN papers in `papers/` get expressivity from few
  qubits. Try with and without.
- Readout: Z expectation on 5 chosen qubits (the centre and four edge
  midpoints) -> 5 numbers -> softmax over classes via a fixed or learned
  linear map. With 18 classes, the linear map is the pragmatic choice.
- Training: PennyLane or Qiskit Machine Learning with the parameter-shift
  rule, Adam, batch 32. Simulator: `default.tensor` (MPS) in PennyLane or
  Aer MPS via Qiskit. Statevector at 25 qubits is 512 MB per state and
  will be too slow for training loops on this machine; MPS with L <= 4
  should be seconds per batch.
- Symmetry: the 5x5 grid has the dihedral group of the square. Either
  augment with the 8 symmetries or build an equivariant ansatz. Augment
  first; it is free.

### What would count as a result

- Rung 4 or 5 beating rung 0 on real misaligned recordings.
- Rung 5 matching rung 2 on input A within a few points. That would mean
  the quantum model, with far fewer parameters, learns the task. Beating
  it would be surprising and would need a careful check for leakage.
- Rung 5 degrading more gracefully than rung 2 as shots per circuit drop.
  That is the noise-robustness claim, and it is testable.
- Any rung correctly flagging `other` on the mug and the hand.

## 6. Evaluation

- Accuracy and confusion matrix on Test 1 and Test 2.
- Accuracy versus jitter, binned in 0.1 zone steps. The template method
  falls off a cliff around 0.3; a learned model should not.
- Accuracy versus shots per circuit for inputs B and C (10 to 500).
- Accuracy versus frames averaged (1 to 30).
- Parameter count and inference cost per decision (circuits, shots).
- For rung 5: repeat on a real backend for a 50-scene subset once the
  simulator numbers are in.

## 7. Implementation steps

1. `dataset.py`: generate synthetic scenes, produce inputs A and B and
   labels, save as `.npz` shards. Parallel over cores. Include the `other`
   generator.
2. `collect_real.py`: guided recording session for the cube shapes, writes
   `data/real/` and `labels.csv`. Reuses `acquire.py`.
3. `train_classical.py`: rungs 1 to 4 with scikit-learn / PyTorch. Report
   Test 1 and Test 2. Half a day once data exists.
4. `train_qnn.py`: rung 5 and 6 with PennyLane. Start with L = 2, no
   re-uploading, 6 classes (square, outline, plus, X, bar, none) to get a
   signal, then widen to 18.
5. `eval.py`: the curves in section 6, one figure each, saved to `results/`.
6. `live.py --model <file>`: run any trained model on the sensor next to
   the template verdict, so both are visible at once.

## 8. Decisions made, and open ones

Made:
- Shape classification first; position, sign and jitter as later heads.
- Synthetic training, real testing. Real data enters training only if the
  synthetic-to-real gap turns out large.
- Nearest-neighbour entanglers only, so the QNN stays simulable and maps
  onto a lattice chip.
- PennyLane for the QNN (autodiff and MPS device out of the box); Qiskit
  stays for the rule-based pipeline and hardware runs.

Open:
- 5x5 with 25 qubits, or 8x8 with 64? 64 qubits is beyond comfortable
  simulation for training. Stay at 25 unless rung 5 is trivially easy.
- Whether to give the model the 8x8 ring for the plateau case, or keep
  that as a classical pre-check. Keep the pre-check for now.
- Class weighting for `other` and `none`, which will dominate real scenes.

## 9. Risks

- The task may be too easy at rung 1, in which case the interesting
  comparison collapses to noise robustness (section 6, shots curve) and
  the `other` class. Plan for that outcome; it is still a result.
- Synthetic-to-real gap in partial-zone behaviour. The simulator's
  straddler model is phenomenological. Mitigation: the ~100 real cube
  recordings, used for validation first and fine-tuning if needed.
- Barren plateaus in rung 5 at 25 qubits. Mitigations: shallow L, local
  cost (few readout qubits), identity-like initialisation, re-uploading.
- Training time. Budget one core-day for rung 5 on MPS before deciding
  whether to shrink to a 3x3 / 9-qubit pilot.

## 10. Environment

- `~/venvs/qlidar` has numpy, qiskit, qiskit-aer, pyserial. Add:
  `pip install pennylane pennylane-lightning torch scikit-learn matplotlib`
- Machine: 32 cores, 123 GB RAM. An NVIDIA GPU is present but `nvidia-smi`
  reports a driver/library version mismatch (NVML 610.57), so it is not
  usable until the driver is fixed. Rung 5 is CPU-bound on MPS contraction
  anyway; 32 cores is plenty for the data generation.
- Directory layout to add: `data/synthetic/`, `data/real/`, `models/`,
  `results/`.

## 11. Literature: machine learning on LiDAR (tracked 2026-09-27)

Organised by data representation, because the representation is the design
choice our project makes. Status column: `-` not read, `skim` abstract
only, `read` in full, `pdf` copy in `papers/`.

PDFs: `papers/lidar-ml/<category>/`, index in `papers/lidar-ml/README.md`.

### Range image (our family: sensor-native 2D grid)

| Paper | Year | What | Status |
|---|---|---|---|
| SqueezeSeg, Wu et al. | 2018 | 2D CNN on spherical range image for segmentation | pdf |
| RangeNet++, Milioto et al. | 2019 | Range-image segmentation with kNN post-processing | pdf |
| LaserNet, Meyer et al. | 2019 | Detection directly on the range image | pdf |
| RSN: Range Sparse Net, Sun et al. | 2021 | Range image for foreground, sparse 3D conv on it. https://arxiv.org/pdf/2106.13365 | pdf |
| RangeFormer, Kong et al. | 2023 | Transformer on range images | pdf |
| Spherical Transformer for LiDAR-Based 3D Recognition | 2023 | Attention in spherical coordinates | pdf |

### Bird's eye view rasters (height field over ground)

| Paper | Year | What | Status |
|---|---|---|---|
| MV3D, Chen et al. | 2017 | Multi-view projections + fusion | pdf |
| PIXOR, Yang et al. | 2018 | BEV occupancy/height channels, single-stage | pdf |
| PointPillars, Lang et al. | 2019 | Learned pillar features -> 2D CNN; deployed workhorse | pdf |
| Voxel or Pillar: Efficient Point Cloud Representation | 2023 | Representation trade-offs. https://arxiv.org/pdf/2304.02867 | pdf |

### Voxel and sparse-voxel transformers (current leaders on nuScenes/Waymo)

| Paper | Year | What | Status |
|---|---|---|---|
| VoxelNet, Zhou and Tuzel | 2018 | Learned voxel features + 3D conv | pdf |
| SECOND, Yan et al. | 2018 | Sparse convolution made voxels practical | - |
| CenterPoint, Yin et al. | 2021 | Anchor-free centre heatmaps; standard baseline | pdf |
| PV-RCNN, Shi et al. | 2020 | Voxel backbone + point refinement | pdf |
| VoTr | 2021 | Voxel transformer with dilated attention | pdf |
| SST: Single Stride Sparse Transformer | 2022 | https://arxiv.org/pdf/2112.06375 | pdf |
| DSVT, FSD | 2023 | Dynamic sparse voxel transformer; fully sparse detector | pdf |
| SparseVoxFormer | 2025 | Sparse voxel transformer, multi-modal. https://arxiv.org/abs/2503.08092 | pdf |
| SV-TransFusion, Sci. Reports | 2026 | Sparse voxel-query interaction. https://www.nature.com/articles/s41598-026-42093-y | pdf |
| Dynamic clustering transformer, Pattern Recognition | 2025 | https://www.sciencedirect.com/science/article/abs/pii/S0031320325011069 | - |

### Point sets and graphs (no grid)

| Paper | Year | What | Status |
|---|---|---|---|
| PointNet / PointNet++, Qi et al. | 2017 | Permutation-invariant per-point features | pdf |
| PointRCNN, 3DSSD, Point Transformer | 2019-21 | Detectors on raw points | pdf |
| Point-GNN, Shi and Rajkumar | 2020 | Graph neural network on points | pdf |

### Fusion, foundation models, generation, robustness (frontier)

| Paper | Year | What | Status |
|---|---|---|---|
| BEVFusion, TransFusion | 2022 | Camera + LiDAR fusion | pdf |
| DeepFusion | 2022 | Lidar/camera/radar modular detector. https://arxiv.org/pdf/2209.12729 | pdf |
| PF3Det | 2025 | Prompted foundation features for LiDAR detection. https://arxiv.org/pdf/2504.03563 | pdf |
| FOMO-3D | 2026 | OWLv2 + Metric3Dv2 priors for long-tail classes. https://arxiv.org/html/2603.08611 | pdf |
| RangeLDM | 2024 | Diffusion model generating realistic range images. https://arxiv.org/pdf/2403.10094 | pdf |
| Comprehensive Robustness Analysis of LiDAR 3D Detection | 2026 | https://arxiv.org/pdf/2607.02074 | pdf |
| Survey on Adversarial Robustness of LiDAR ML Perception | 2024 | https://arxiv.org/pdf/2411.13778 | pdf |
| LiFT: FPGA-tailored 3D detection | 2025 | Small-compute regime. https://arxiv.org/pdf/2501.11159 | pdf |

### Surveys

| Paper | Year | Link | Status |
|---|---|---|---|
| Survey of LiDAR-Based Monomodal 3D Object Detection, Liu and Wang | 2026 | https://link.springer.com/chapter/10.1007/978-981-95-7660-9_3 | - |
| Survey on Deep-Learning-Based LiDAR 3D Object Detection for AD | 2022 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9784304/ | - |
| Comprehensive survey of LiDAR 3D detection with deep learning | 2021 | https://www.sciencedirect.com/science/article/abs/pii/S0097849321001321 | - |
| Review of 3D Object Detection with Vision-Language Models | 2025 | https://arxiv.org/pdf/2504.18738 | pdf |

### Pre-deep-learning (hand-crafted features + SVM / random forest)

Height statistics, planarity, intensity histograms into SVMs or random
forests; still standard in airborne survey ground classification and
full-waveform work (wavelet SVM, PMC6679236). Relevant because our 40 edge
strengths are a hand-crafted feature vector of the same kind.

### Quantum ML on LiDAR

| Paper | Year | What | Status |
|---|---|---|---|
| Hybrid quantum-classical 3D object detection, MC-QCNN, J. Supercomputing | 2025 | The only QML result on real LiDAR (KITTI). Paywalled. DOI 10.1007/s11227-025-06968-7 | - |
| Everything in `papers/quantum-encoding-point-clouds-and-fields/` | 2017-26 | Encodings, quantum edge detection, point-cloud QNNs; none on LiDAR | pdf |
| Goyal, Uehara, Spanias: per-pixel RY + destructive swap edge detection on hardware | 2026 | Closest circuit to ours, on histopathology. https://arxiv.org/abs/2606.21752 | skim |

### Reading order for this plan

1. PointPillars and SqueezeSeg: the two representations closest to ours,
   and the model sizes the field considers "small".
2. RSN: range image as the place to decide where to look, which is what
   our auto-crop does by hand.
3. LiFT: what small-compute LiDAR detection looks like when it is taken
   seriously; the regime our 25-qubit model actually competes in.
4. RangeLDM: if we ever want a learned scene generator instead of
   `sensor_sim`.
5. The 2026 robustness analysis: the evaluation axes (sparsity, noise,
   weather) our shots-and-frames curves should mirror.
