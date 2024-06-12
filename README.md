# Deep-Unfolded-Hybrid-Beamforming-in-Reconfigurable-Intelligent-Surface-MIMO-OFDM-Systems
- Official repository of [Deep Unfolded Hybrid Beamforming in Reconfigurable Intelligent Surface Aided mmWave MIMO-OFDM Systems]([https://ieeexplore.ieee.org/document/9860799](https://ieeexplore.ieee.org/document/10422799))
- This repository contains the code and the dataset
# About
## Contents:
- **sparse_SV_channel_RIS**
  - Testing data (`sparse_SV_channel_RIS.py`)
  - Training data (`sparse_SV_channel_RIS_slices.py`)
- **RIS_WMMSE-MO**
- **RIS_DU** (deep-unfolded version)
- **Benchmark** (GMD-PCA, T-WMMSE-MO, T-SVD)

## Usage Instructions:

### 1. Generate Testing/Training Data
1. Open the following scripts:
   - Testing data: `sparse_SV_channel_RIS.py`
   - Training data: `sparse_SV_channel_RIS_slices.py`
2. Modify the storage location and basic parameters for Testing/Training data as needed.
3. Run the scripts.

### 2. RIS_WMMSE-MO
1. Set up the Python environment (see the list at the end).
2. Open the file `RIS_WMMSE_MO.py`.
3. Modify the storage location for Testing data and basic parameters (N_phi, Nt, Nr, Ns, Nrf_t, Nrf_r, Mt, Mr, Nk).
4. Update iteration counts:
   - Outer iteration: change the value of variable "Iter".
   - Inner iteration: open `solver.py` (`.\benson_code\RIS_WMMSE_MO\mypymanopt\solvers`) and change the `maxiter` value.
5. Save the changes and run `RIS_WMMSE_MO.py`.

### 3. RIS_DU
1. Set up the Python environment (see the list at the end).
2. Open the file `RIS_DU_train_batch.py` (Training phase).
3. Modify the storage location for Training data, training weights, and loss graph, as well as basic parameters (e.g., `Io` for outer iteration count, `In` for inner iteration count).
4. Update iteration counts:
   - Outer iteration: control using comments (`#`). For example, to change from 6 to 5 layers, comment out lines 708 and 709.
   - Inner iteration (FRF, RIS, WRF): control using comments (`#`). For example, comment out lines 291 and 330 for 4-layer iterations of F_RF, comment out lines 439 and 479 for 4-layer iterations of RIS element, and comment out lines 572 and 607 for 4-layer iterations of W_RF
5. Run `RIS_DU_train_batch.py`.
6. Open the file `RIS_DU_x_x_test.py` (Testing phase).
7. Modify the storage location for Testing data, training weights, and basic parameters (e.g., `Io` for outer iteration count, `In` for inner iteration count).
8. Run the script to obtain test results.

### 4. Benchmark-GMD_PCA
1. Open the file `GMD_PCA.m`.
2. Modify the storage location for Testing data and basic parameters.
3. Run the script.

### 5. Benchmark-T_SVD
1. Open the file `T_SVD_demo.m`.
2. Modify the storage location for Testing data and basic parameters.
3. Run the script.
4. If there are issues with missing `.m` files, locate the files in `sub_func_test_ca` and add the path using MATLAB (`Home -> Set Path`). If files are missing, find them in `T-SVD-BF.zip`.

### 6. Benchmark-T_WMMSE_MO
1. Set up the Python environment (see the list at the end).
2. Open the file `T_SVD_demo.m`.
3. Modify the storage location for Testing data and basic parameters.
4. Update iteration counts similar to `RIS_WMMSE_MO`.
   - Note: inner iteration solver file should be opened from `T_WMMSE_MO` (`.\benson_code\Benchmark\T-WMMSE-MO\mypymanopt\solvers`).
5. Run the script.

## Hardware Specifications:
| ----- | --------------------- |
| CPU   |  Intel Core i7-12700  |
| RAM   |    DDR4-3200 64GB     |
| GPU   |   Nvidia RTX 3060 Ti  |

### Python Environment:
| Name             | Version |
| ---------------- | ------- |
| python           | 3.9.17  |
| tensorflow       | 2.6.0   |
| tensorflow-gpu   | 2.6.0   |
| keras            | 2.6.0   |
| numpy            | 1.25.2  |
| matplotlib       | 3.5.3   |
| scipy            | 1.11.1  |

## Citation
```
@ARTICLE{10422799,
  author={Chen, Kuan-Ming and Chang, Hsin-Yuan and Chang, Ronald Y. and Chung, Wei-Ho},
  journal={IEEE Wireless Communications Letters}, 
  title={Deep Unfolded Hybrid Beamforming in Reconfigurable Intelligent Surface Aided mmWave MIMO-OFDM Systems}, 
  year={2024},
  month=apr,
  volume={13},
  number={4},
  pages={1118-1122}}
```
