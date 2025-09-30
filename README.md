# Backtoseries API — Machine Learning forecasting for Time Series

This repository contains an API applying Machine Learning to economic time series, with components for data collection and preprocessing, experiments, backtesting, and an API to serve forecasts.

---

## 📂 Project Structure

```text
projfinal-ml/
│
├── app/
│   ├── main.py                # Application / API entry point
│   ├── api/                   # API endpoints (routes, controllers)
│   ├── models/                # Trained models, pipelines, serialization
│   ├── data/                  # Data ingestion, transformation, cleaning
│   ├── backtesting/           # Simulations, historical tests, validations
│   └── utils/                 # Utility functions, metrics, helpers
│
├── notebooks/                 # Analysis and experiment notebooks (optional)
├── tests/                     # Unit / integration tests (if available)
├── requirements.txt           # Project dependencies
├── .gitignore
└── README.md                  # This file
```

---

## 🎯 Objective

The goal of this project is to:

- apply **machine learning** techniques to economic time series data;
- perform **backtesting** to evaluate historical model performance;
- serve forecasts through a **web API** for external consumption;
- compare different algorithms (regression, tree-based models, neural networks, etc.);
- generate visualizations and performance reports.

---

## 🧰 Technologies Used

- **Python** (3.x)
- Core libraries: `pandas`, `numpy`, `scikit-learn`
- Web framework: **FastAPI**
- ASGI server: **Uvicorn**
- Visualization: `matplotlib`, `seaborn`, `plotly`, etc.

---

## 📦 API Endpoints (examples)

Here are examples of endpoints that should be documented:

| Method | Route                | Description                                   |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/predict?period=30`  | Returns the forecast for the next 30 days     |
| POST   | `/train`              | Triggers model retraining                     |
| GET    | `/metrics`            | Returns performance metrics                   |
| ...    | ...                   | Other project-specific endpoints              |

---

## 🔍 Experiments & Backtesting

- **Features / variables** used as model inputs  
- **Preprocessing**: normalization, handling missing values, creation of lags, rolling windows  
- **Models compared**: linear regression, decision trees, random forest, neural networks, etc.  
- **Validation strategy**: cross-validation, rolling windows, walk-forward, etc.  
- **Backtesting**: historical simulation, evaluation of hypothetical profitability  
- **Metrics**: MAE (Mean Absolute Error), RMSE, MAPE, simulated financial return  
