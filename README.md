# AETHER-P³  
**Accelerometer-driven Estimation of THERmospheric Density – A Physics-Informed Probabilistic Prediction Platform**

## Overview
AETHER-P³ is a machine-learning-based framework for **global thermospheric mass density forecasting with uncertainty quantification**.  
The model is designed for **multi-step forecasting** (up to 6 hours ahead) using recent space weather conditions, satellite orbital information, and baseline empirical model estimates.

The forecasting problem is formulated as a **sequence-to-sequence regression task**, and predictive uncertainty is estimated using **Evidential Deep Learning (Normal-Gamma output)**.

This repository contains the **training, inference, and evaluation pipeline** used in our Space Weather Journal submission.

---

## Key Features
- Multi-step thermospheric density forecasting  
- Evidential Deep Learning with Normal-Gamma uncertainty output  
- Physics-informed inputs using empirical models (JB2008, NRLMSISE-00)  
- Robust evaluation across quiet, moderate, and extreme geomagnetic conditions  
- Multi-seed training and uncertainty reliability assessment  

---

## Model Architecture
- Dual recurrent encoders (LSTM-based)
- Sequence-to-sequence forecasting framework
- Evidential Normal-Gamma output head

---

## Data Sources
All data used in this project are **publicly available**:

- **Satellite accelerometer data**  
  CHAMP, GRACE, GOCE, SWARM-C

- **Empirical density models**  
  - JB2008  
  - NRLMSISE-00  

- **Solar & geomagnetic indices**
  - F10.7, F10.7A, F30  
  - Dst, Apo30
  - Solar wind parameters (Bz, velocity, proton density, AE)

⚠️ **Note:** Due to data volume and licensing, raw datasets are **not included** in this repository.

---

## Citation

If you use this code or build upon this work, please cite:

Wang, R., Bai, X., et al.  
A Machine-Learning-Based Global Thermospheric Density Forecasting Model  
Space Weather, under review.

