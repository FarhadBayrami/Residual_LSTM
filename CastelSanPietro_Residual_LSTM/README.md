# Residual LSTM for Hydrological Discharge Forecasting

## Project Overview

This project implements a **Hybrid Residual Correction LSTM** model to improve river discharge predictions. It uses the physics-based **TOPKAPI** model as a baseline and trains an LSTM to learn and correct the residual error (`Observed QM - TOPKAPI Q`).

**Basin**: Casalecchio, Italy  
**Time Period**: 2013 – 2026 (Hourly data)

---

## Performance Results

| Model                    | MAE (m³/s) | RMSE (m³/s) | Improvement vs TOPKAPI |
|--------------------------|------------|-------------|------------------------|
| **Residual LSTM**        | **8.63**   | **16.97**   | **+25.7%**             |
| TOPKAPI (Baseline)       | 11.62      | 19.99       | -                      |
| Pure TCN                 | 8.33       | 17.23       | -                      |

---

## Project Structure

```bash
hydrology_residual_lstm/
├── data/                     # Raw data (not uploaded - large file)
│   └── 425.sbs.ts
├── results/                  # Prediction CSVs and plots
├── models/                   # Trained model & scalers
├── scripts/
│   ├── data_preparation_residual.py
│   ├── train_residual_lstm.py
│   └── evaluate_residual.py
├── README.md
├── requirements.txt
└── .gitignore