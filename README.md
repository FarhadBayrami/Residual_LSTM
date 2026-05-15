# Residual LSTM for Hydrological Discharge Forecasting

## Project Description

Hybrid **Residual Correction LSTM** model that improves the physics-based **TOPKAPI** model by learning the residual error (`Observed - Predicted`).

**Basin**: Casalecchio  
**Time Resolution**: Hourly (2013–2026)

---

## Performance Results

| Model                    | MAE (m³/s) | RMSE (m³/s) | Improvement |
|--------------------------|------------|-------------|-------------|
| **Residual LSTM**        | **8.63**   | **16.97**   | **+25.7%**  |
| TOPKAPI (Baseline)       | 11.62      | 19.99       | -           |
| Pure TCN                 | 8.33       | 17.23       | -           |

---

## Project Structure

```bash
hydrology_residual_lstm/
├── data/                     # Raw data (not included - large file)
│   └── 425.sbs.ts
├── results/                  # CSV predictions and plots
├── models/                   # Trained model & scalers
├── scripts/
│   ├── data_preparation_residual.py
│   ├── train_residual_lstm.py
│   └── evaluate_residual.py
├── README.md
├── requirements.txt
└── .gitignore
