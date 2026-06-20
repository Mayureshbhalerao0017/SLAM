# 🗺️ SLAM Core Library 

![C++](https://img.shields.io/badge/C++-17-blue)
![Eigen](https://img.shields.io/badge/Eigen-3.4-orange)
![GTSAM](https://img.shields.io/badge/GTSAM-4.2-green)

A pure C++ mathematical library for 2D autonomous vehicle state estimation, mapping, and loop closure. This repository strips away all middleware (like ROS/ROS 2) to provide a lightweight, hyper-fast, and highly portable suite of Simultaneous Localization and Mapping (SLAM) algorithms.

This library is designed using the **Wrapper Pattern**, meaning it can be directly compiled into embedded systems, bare-metal microcontrollers, or any proprietary autonomous driving stack via custom wrappers.

## 📦 Core Algorithms

This repository implements three distinct SLAM paradigms, allowing users to choose the right algorithm based on computational constraints and accuracy requirements.

### 1. EKF SLAM (`EKFSLAMCore.hpp`)
* **Paradigm:** Extended Kalman Filter (Feature-based).
* **Features:** Dynamically resizing state and covariance matrices, non-holonomic kinematic updates, and Mahalanobis-distance data association gates. Highly computationally efficient.

### 2. FastSLAM 2.0 (`FastSLAMCore.hpp`)
* **Paradigm:** Rao-Blackwellized Particle Filter.
* **Features:** Maintains a highly diverse hypothesis space (default 30 particles). Features systematic resampling to prevent weight collapse and dynamic kinematic noise injection. Excellent for handling severe data-association ambiguities.

### 3. GraphSLAM (`GraphSLAMCore.hpp`)
* **Paradigm:** Factor Graph Optimization (via `GTSAM`).
* **Features:** Uses Incremental Smoothing and Mapping (iSAM2) for real-time performance. Incorporates Huber-loss robust noise models to gracefully absorb massive coordinate corrections during lap-2 loop closures without mathematical tearing.

---

## 🏗️ Data Architecture

To ensure maximum portability, the library relies on standard C++ vectors and custom structs defined in `SLAMDataTypes.hpp`:

* **`Observation`**: Generic sensor inputs (e.g., from LiDAR, Camera) containing `range`, `bearing`, and a custom `class_id` (e.g., left boundary, right boundary, loop-closure anchor).
* **`Pose2D`**: Mathematical 2D coordinate frames (`x`, `y`, `yaw`).
* **`Landmark`**: The mapped environmental features, including probabilistic confidence scores.

---

## ⚙️ Prerequisites

* **C++ Compiler:** C++17 or higher
* **Build System:** CMake (3.10+)
* **Math Libraries:** * [Eigen3](https://eigen.tuxfamily.org/) (Linear Algebra)
  * [GTSAM](https://gtsam.org/) (Factor Graphs)

**Ubuntu 22.04 Setup:**
```bash
sudo apt-get update
sudo apt-get install cmake libeigen-dev software-properties-common
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt-get install libgtsam-dev libgtsam-unstable-dev
