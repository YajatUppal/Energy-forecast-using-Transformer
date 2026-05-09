# Energy Demand Forecasting using Patch Transformers

## Project Overview

This project implements a Patch Transformer model to forecast electricity demand from historical time-series data, and uses a LSTM model as a baseline. The goal is to predict the next 24 hours of energy consumption using previous energy usage patterns and time-based features.

These 2 models are built and compared:

1. LSTM baseline model
2. Patch Transformer model

The Patch Transformer improves over a standard Transformer by dividing long time-series sequences into smaller temporal patches before applying self-attention.

## Datasets

Two real-world electricity consumption datasets were used.

### PJM Energy Consumption Dataset

The primary dataset is the PJM hourly energy consumption dataset for the PJME region.

- Target variable: `PJME_MW`
- Frequency: hourly
- Input window: 336 hours
- Forecast horizon: 24 hours

Features:

- `PJME_MW`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`

### Tetouan City Power Consumption Dataset

A second dataset from Tetouan, Morocco was used for validation.

- Target variable: `PowerConsumption_Zone1`
- Frequency: 10-minute intervals
- Input window: 336 time steps
- Forecast horizon: 24 time steps

Features:

- `Temperature`
- `Humidity`
- `WindSpeed`
- `GeneralDiffuseFlows`
- `DiffuseFlows`
- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`

## Model Architectures

### LSTM Baseline

The LSTM baseline consists of:

- One LSTM layer with 32 hidden units
- One dense output layer
- Adam optimizer
- MSE loss
- MAE metric

### Patch Transformer

The Patch Transformer consists of:

- Patch extraction layer
- Dense patch embedding
- Positional embedding
- Two Transformer encoder blocks
- Multi-head self-attention
- Attention pooling
- Dense forecasting layer

For the PJM dataset, the input sequence of 336 hours is divided into 12-hour patches, resulting in 28 patch tokens.

## Project Structure

```text
DL-energy-forecast/
│
├── data/
│   ├── PJME_hourly.csv
│   └── powerconsumption.csv
│
├── figures/
│   └── saved plots and comparison graphs
│
├── models/
│   └── saved trained models and weights
│
├── notebooks/
│   └── exploration.ipynb
│
├── report/
│   └── final report and slide deck
│
├── src/
│   ├── preprocessing.py
│   ├── preprocessing_tetouan.py
│   ├── make_windows.py
│   ├── make_windows_tetouan.py
│   ├── lstm.py
│   ├── lstm_tetouan.py
│   ├── transformer.py
│   ├── transformer_tetouan.py
│   ├── plot_comparison.py
│   ├── plot_comparison_tetouan.py
│   ├── graphs.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt