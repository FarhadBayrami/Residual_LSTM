<div align="center">

# 🌊 Residual LSTM for Hydrological Discharge Forecasting
### Hybrid Physics-ML Correction Model for the CastelSanPietro Basin

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/MAE-8.63%20m³%2Fs-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/RMSE-16.97%20m³%2Fs-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Improvement-25.7%25%20over%20TOPKAPI-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Resolution-Hourly%202013--2026-red?style=flat-square"/>
</p>

*A hybrid deep learning model that learns the residual error of the physics-based TOPKAPI hydrological model, significantly improving river discharge forecasting accuracy.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Approach](#-approach)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)

---

## 🔬 Overview

Physics-based hydrological models like **TOPKAPI** are grounded in physical laws but systematically accumulate errors due to model uncertainty, parameter estimation issues, and unrepresented processes. This project addresses that gap with a **Residual Correction LSTM** — a neural network trained not to replace the physics model, but to learn and correct its errors.

The key insight: train an LSTM on the residual `(Observed discharge − TOPKAPI predicted discharge)`, then add the correction back to the physics model output at inference time.

---

## ⚙️ Approach
Observed Discharge
│
  ▼
Residual = Observed − TOPKAPI Predicted
│
▼
┌─────────────┐
│  LSTM Model │  ← Learns temporal patterns in the error signal
└──────┬──────┘
│  Predicted Residual
▼
Final Forecast = TOPKAPI + Predicted Residual
This hybrid approach preserves the physical interpretability of TOPKAPI while leveraging deep learning to correct systematic biases.

---

## 📦 Dataset

| Property        | Value                          |
|-----------------|-------------------------------|
| Basin           | CastelSanPietro               |
| Time resolution | Hourly                        |
| Time span       | 2013 – 2026                   |
| Data file       | `425.sbs.ts` (raw, not included — large file) |
| Target variable | River discharge (m³/s)        |

> ⚠️ Raw data is not included in this repository due to file size. Place `425.sbs.ts` in the `CastelSanPietro_Residual_LSTM/data/` folder before running.

---

## 🧠 Model Architecture

| Component       | Details                                 |
|-----------------|-----------------------------------------|
| Architecture    | Stacked LSTM (Residual Correction)      |
| Input           | TOPKAPI predictions + meteorological features |
| Target          | Residual error (Observed − TOPKAPI)     |
| Loss Function   | Mean Squared Error (MSE)                |
| Optimizer       | Adam                                    |
| Framework       | PyTorch                                 |

---

## 📊 Results

Evaluated on the CastelSanPietro basin test set:

| Model | MAE (m³/s) | RMSE (m³/s) | vs. Baseline |
|-------|-----------|------------|--------------|
| **Residual LSTM (ours)** | **8.63** | **16.97** | **+25.7%** |
| TOPKAPI (physics baseline) | 11.62 | 19.99 | — |
| Pure TCN | 8.33 | 17.23 | — |

> The Residual LSTM achieves a **25.7% reduction in MAE** over the physics-only baseline, demonstrating the value of hybrid physics-ML approaches for hydrological forecasting.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
# 1. Clone the repository
git clone https://github.com/FarhadBayrami/Residual_LSTM.git
cd Residual_LSTM/CastelSanPietro_Residual_LSTM

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw data file (425.sbs.ts) in the data/ folder

# 4. Prepare data
python scripts/data_preparation_residual.py

# 5. Train the model
python scripts/train_residual_lstm.py

# 6. Evaluate
python scripts/evaluate_residual.py
```

---

## 📁 Project Structure

**`Residual_LSTM/`**

| Path | Description |
|------|-------------|
| `CastelSanPietro_Residual_LSTM/data/` | Raw data — not included (large file) |
| `CastelSanPietro_Residual_LSTM/data/425.sbs.ts` | Raw discharge time series |
| `CastelSanPietro_Residual_LSTM/results/` | CSV predictions and plots |
| `CastelSanPietro_Residual_LSTM/models/` | Saved model weights and scalers |
| `CastelSanPietro_Residual_LSTM/scripts/data_preparation_residual.py` | Data loading and preprocessing |
| `CastelSanPietro_Residual_LSTM/scripts/train_residual_lstm.py` | Model training |
| `CastelSanPietro_Residual_LSTM/scripts/evaluate_residual.py` | Evaluation and visualisation |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
## 🔮 Future Work

- [ ] Extend to multi-basin generalisation (transfer learning)
- [ ] Incorporate rainfall forecast uncertainty (ensemble inputs)
- [ ] Compare with Transformer-based sequence models
- [ ] Real-time operational deployment pipeline
- [ ] Explainability: SHAP values for feature importance in residual correction

---

## 📚 References

1. Liu, Y. et al. — *Improving Hydrological Model Runoff Estimation with Deep Learning*, Hydrology and Earth System Sciences, 2021.
2. Kratzert, F. et al. — *Rainfall–runoff modelling using Long Short-Term Memory (LSTM)*, HESS, 2018.
3. Todini, E. — *The TOPKAPI model*, Journal of Hydrology, 1996.
4. Hochreiter, S. & Schmidhuber, J. — *Long Short-Term Memory*, Neural Computation, 1997.

---

## 👤 Author

**Farhad Bayrami**
MSc Student — University of Bologna
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

---

<div align="center">
  <sub>Built with ❤️ as part of a Hydrology & Machine Learning research project at the University of Bologna</sub>
</div>
