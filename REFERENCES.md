# References

Papers are cited here, not redistributed. Links point to the authors'
open-access versions where they exist.

## Quantum image encoding and quantum edge detection

1. X.-W. Yao, H. Wang, Z. Liao, et al. *Quantum Image Processing and Its Application to Edge Detection: Theory and Experiment.* Phys. Rev. X 7, 031041 (2017). [arXiv:1801.01465](https://arxiv.org/abs/1801.01465)
   The origin of QPIE amplitude encoding and Quantum Hadamard Edge Detection (QHED).
2. *Quantum Image Processing: A Comparative Study of NEQR and FRQI Encoding Schemes with Hybrid Processing.* GLSVLSI 2025. [doi:10.1145/3716368.3735286](https://doi.org/10.1145/3716368.3735286)
3. F. Sun. *A Fully Quantum Algorithm for Image Edge Detection.* 2026. [arXiv:2604.23535](https://arxiv.org/abs/2604.23535)
4. M. A. Sohail, G. Pinheiro, Y. Poyraz Koçak, et al. *Quantum Gradient-Based Approach for Edge and Corner Detection Using Sobel Kernels.* 2026. [arXiv:2605.00744](https://arxiv.org/abs/2605.00744)
5. N. Goyal, G. Uehara, A. Spanias. *Configurable Algorithms for Histopathologic Cancer Detection on Quantum Hardware.* 2026. [arXiv:2606.21752](https://arxiv.org/abs/2606.21752)
   Per-pixel RY encoding with destructive-swap edge detection on hardware; the circuit closest to ours.
6. Y. Zhang, K. Lu, Y. Gao. *QSobel: A novel quantum image edge extraction algorithm.* Sci. China Inf. Sci. 58 (2015).
7. A. Senokosov, A. Sedykh, A. Sagingalieva, B. Kyriacou, A. Melnikov. *Quantum machine learning for image classification.* Mach. Learn.: Sci. Technol. 5, 015040 (2024). [doi:10.1088/2632-2153/ad2aef](https://doi.org/10.1088/2632-2153/ad2aef), [arXiv:2304.09224](https://arxiv.org/abs/2304.09224)
   Two hybrid models on MNIST: parallel quantum circuits (99.21% with 8x fewer parameters than the classical equivalent) and a quanvolutional layer that matches its classical counterpart with 4x fewer trainable parameters. A reference point for the quanvolution plan.

## Quantum machine learning on point clouds

8. J. Heredge, C. Hill, L. Hollenberg, M. Sevior. *Permutation Invariant Encodings for Quantum Machine Learning with Point Cloud Data.* 2023. [arXiv:2304.03601](https://arxiv.org/abs/2304.03601)
9. L. Rathi, E. Tretschk, C. Theobalt, R. Dabral, V. Golyanik. *3D-QAE: Fully Quantum Auto-Encoding of 3D Point Clouds.* BMVC 2023. [arXiv:2311.05604](https://arxiv.org/abs/2311.05604)
10. Z. Li, L. Nagano, K. Terashi. *Enforcing exact permutation and rotational symmetries in the application of quantum neural network on point cloud datasets.* Phys. Rev. Research 6, 043028 (2024). [arXiv:2405.11150](https://arxiv.org/abs/2405.11150)
11. N. Kuete Meli, J. Lukasik, V. Golyanik, M. Moeller. *Layered Quantum Architecture Search for 3D Point Cloud Classification.* 2026. [arXiv:2603.20024](https://arxiv.org/abs/2603.20024)
12. *HyQuRP: Hybrid quantum-classical neural network with rotational and permutational equivariance for 3D point clouds.* 2026. [arXiv:2602.06381](https://arxiv.org/abs/2602.06381)
13. F. Fan, et al. *Quantum Circuit-Based Learning Models: Bridging Quantum Computing and Machine Learning* (survey). 2026. [arXiv:2602.00048](https://arxiv.org/abs/2602.00048)
14. *Hybrid quantum-classical 3D object detection using multi-channel quantum convolutional neural network.* J. Supercomputing (2025). [doi:10.1007/s11227-025-06968-7](https://doi.org/10.1007/s11227-025-06968-7)
    The only QML result we found on real LiDAR data (KITTI). It works on Cartesian points, not the range image.
15. A. Frangou, S. Chretien, I. Rungger. *The First Quantum Co-processor Hybrid for Processing Quantum Point Cloud Multimodal Sensor Data.* Proc. FTC 2019, Springer. [doi:10.1007/978-3-030-32520-6_32](https://doi.org/10.1007/978-3-030-32520-6_32)
16. *A flexible quantum point cloud representation supporting attribute filtering and downsampling.* Physica Scripta (2025). [doi:10.1088/1402-4896/ae1db5](https://doi.org/10.1088/1402-4896/ae1db5)

## Quantum kernel methods

17. C. Ding, S. Wang, Y. Wang, W. Gao. *Quantum machine learning for multiclass classification beyond kernel methods.* Phys. Rev. A 111, 062410 (2025). [arXiv:2411.02913](https://arxiv.org/abs/2411.02913)
    Extends the Havlíček et al. quantum-feature-space classifier from binary to multiclass, comparing six quantum kernels against classical methods on real-world datasets. Relevant to a kernel SVM on our encoding, which needs one-vs-rest or a native multiclass method for the 17 shape classes.

## Classical machine learning on LiDAR

### Range image (the sensor-native 2-D grid, the representation this project uses)

18. B. Wu, A. Wan, X. Yue, K. Keutzer. *SqueezeSeg.* ICRA 2018. [arXiv:1710.07368](https://arxiv.org/abs/1710.07368)
19. A. Milioto, I. Vizzo, J. Behley, C. Stachniss. *RangeNet++.* IROS 2019.
20. G. P. Meyer, et al. *LaserNet.* CVPR 2019. [arXiv:1903.08701](https://arxiv.org/abs/1903.08701)
21. P. Sun, et al. *RSN: Range Sparse Net.* CVPR 2021. [arXiv:2106.13365](https://arxiv.org/abs/2106.13365)
22. L. Kong, et al. *Rethinking Range View Representation for LiDAR Segmentation (RangeFormer).* ICCV 2023. [arXiv:2303.05367](https://arxiv.org/abs/2303.05367)
23. X. Lai, et al. *Spherical Transformer for LiDAR-based 3D Recognition.* CVPR 2023. [arXiv:2303.12766](https://arxiv.org/abs/2303.12766)

### Bird's-eye view

24. X. Chen, et al. *MV3D: Multi-View 3D Object Detection.* CVPR 2017. [arXiv:1611.07759](https://arxiv.org/abs/1611.07759)
25. B. Yang, W. Luo, R. Urtasun. *PIXOR.* CVPR 2018. [arXiv:1902.06326](https://arxiv.org/abs/1902.06326)
26. A. H. Lang, et al. *PointPillars.* CVPR 2019. [arXiv:1812.05784](https://arxiv.org/abs/1812.05784)
27. Y. Huang, S. Zhou, J. Zhang, J. Dong, N. Zheng. *Voxel or Pillar: Exploring Efficient Point Cloud Representation for 3D Object Detection.* 2023. [arXiv:2304.02867](https://arxiv.org/abs/2304.02867)

### Voxel and sparse-voxel models

28. Y. Zhou, O. Tuzel. *VoxelNet.* CVPR 2018. [arXiv:1711.06396](https://arxiv.org/abs/1711.06396)
29. Y. Yan, Y. Mao, B. Li. *SECOND: Sparsely Embedded Convolutional Detection.* Sensors 18(10), 3337 (2018).
30. S. Shi, et al. *PV-RCNN.* CVPR 2020. [arXiv:1912.13192](https://arxiv.org/abs/1912.13192)
31. T. Yin, X. Zhou, P. Krähenbühl. *CenterPoint.* CVPR 2021. [arXiv:2006.11275](https://arxiv.org/abs/2006.11275)
32. J. Mao, Y. Xue, et al. *Voxel Transformer (VoTr).* ICCV 2021. [arXiv:2109.02497](https://arxiv.org/abs/2109.02497)
33. L. Fan, et al. *SST: Single-Stride Sparse Transformer.* CVPR 2022. [arXiv:2112.06375](https://arxiv.org/abs/2112.06375)
34. H. Wang, et al. *DSVT: Dynamic Sparse Voxel Transformer.* CVPR 2023. [arXiv:2301.06051](https://arxiv.org/abs/2301.06051)
35. L. Fan, F. Wang, N. Wang, Z. Zhang. *Fully Sparse 3D Object Detection (FSD).* NeurIPS 2022. [arXiv:2207.10035](https://arxiv.org/abs/2207.10035)
36. H. Son, et al. *SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection.* 2025. [arXiv:2503.08092](https://arxiv.org/abs/2503.08092)
37. T. Shi. *SV-TransFusion for LiDAR 3D object detection with Sparse Voxel-Query Interaction.* Scientific Reports (2026).

### Point sets and graphs

38. C. R. Qi, H. Su, K. Mo, L. J. Guibas. *PointNet.* CVPR 2017. [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
39. C. R. Qi, L. Yi, H. Su, L. J. Guibas. *PointNet++.* NeurIPS 2017. [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)
40. S. Shi, X. Wang, H. Li. *PointRCNN.* CVPR 2019. [arXiv:1812.04244](https://arxiv.org/abs/1812.04244)
41. W. Shi, R. Rajkumar. *Point-GNN.* CVPR 2020. [arXiv:2003.01251](https://arxiv.org/abs/2003.01251)
42. Z. Yang, et al. *3DSSD.* CVPR 2020. [arXiv:2002.10187](https://arxiv.org/abs/2002.10187)
43. H. Zhao, et al. *Point Transformer.* ICCV 2021. [arXiv:2012.09164](https://arxiv.org/abs/2012.09164)

### Fusion, generation, robustness, small compute

44. X. Bai, et al. *TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers.* CVPR 2022. [arXiv:2203.11496](https://arxiv.org/abs/2203.11496)
45. Z. Liu, H. Tang, et al. *BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.* 2022. [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)
46. F. Drews, D. Feng, F. Faion, L. Rosenbaum, M. Ulrich, C. Gläser. *DeepFusion: A Robust and Modular 3D Object Detector for Lidars, Cameras and Radars.* 2022. [arXiv:2209.12729](https://arxiv.org/abs/2209.12729)
47. Q. Hu, Z. Zhang, W. Hu. *RangeLDM: Fast Realistic LiDAR Point Cloud Generation.* 2024. [arXiv:2403.10094](https://arxiv.org/abs/2403.10094)
48. J. Kim, A. Kaur. *A Survey on Adversarial Robustness of LiDAR-based Machine Learning Perception in Autonomous Vehicles.* 2024. [arXiv:2411.13778](https://arxiv.org/abs/2411.13778)
49. K. Lis, T. Kryjak, M. Gorgoń. *LiFT: Lightweight, FPGA-tailored 3D object detection based on LiDAR data.* 2025. [arXiv:2501.11159](https://arxiv.org/abs/2501.11159)
50. K. Li, T. Zhang, K.-C. Peng, G. Wang. *PF3Det: A Prompted Foundation Feature Assisted Visual LiDAR 3D Detector.* 2025. [arXiv:2504.03563](https://arxiv.org/abs/2504.03563)
51. R. Sapkota, et al. *A Review of 3D Object Detection with Vision-Language Models.* 2025. [arXiv:2504.18738](https://arxiv.org/abs/2504.18738)
52. A. J. Yang, J. Tu, et al. *FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection.* 2026. [arXiv:2603.08611](https://arxiv.org/abs/2603.08611)
53. A. Chandorkar, et al. *Comprehensive Robustness Analysis of LiDAR-based 3D Object Detection in Autonomous Driving.* 2026. [arXiv:2607.02074](https://arxiv.org/abs/2607.02074)

## Sensor documentation

- DFRobot SEN0628 product page and wiki: https://www.dfrobot.com/product-2999.html, https://wiki.dfrobot.com/sen0628/
- STMicroelectronics VL53L7CX datasheet: https://www.st.com/en/imaging-and-photonics-solutions/vl53l7cx.html

## Software

- Qiskit and Qiskit Aer: https://github.com/Qiskit
- NumPy, pySerial
