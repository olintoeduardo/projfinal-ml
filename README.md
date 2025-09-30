# Backtoseries API — Machine Learning forecasting for Time Series

This repository contains an API applying Machine Learning to economic time series, with components for data collection and preprocessing, experiments, backtesting, and an API to serve forecasts.

---

## 📂 Project Structure

```text
├── README.md
├── app
│   ├── api
│   │   ├── v1
│   │   │   ├── __init__.py
│   │   │   ├── api.py
│   │   │   └── endpoints
│   │   │       ├── __pycache__
│   │   │       ├── backtest.py
│   │   │       ├── datasets.py
│   │   │       └── transformations.py
│   │   └── v2
│   ├── core
│   │   └── config.py
│   ├── main.py
│   ├── schemas
│   │   ├── backtest.py
│   │   ├── datasets.py
│   │   └── transformations.py
│   └── services
│       └── backtest
│           └── backtest_service.py
├── requirements.txt
└── uploads
    └── 0__cumsum.csv
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

## 📦 API Endpoints

### Dataset
- Users can interact with datasets they wish to model.  
- Must be able to upload `.csv` files containing time series data (via **POST** requests to the API).  
- Uploaded files are mapped to a dedicated folder in the GitHub repository (future plan: migrate to a SQL server).  
- Users can issue **GET** requests to preview basic dataset information (e.g., data samples), using a unique identifier.  

---

### Transformation
Users can apply simple data transformations using common Pandas functions:
- **Percentage Change**: calculates relative changes over consecutive periods; useful for analyzing short- and long-term trends.  
- **Difference**: computes the difference between consecutive periods; essential for making series stationary.  
- **Moving Average**: smooths data over a rolling window; reduces short-term noise and reveals patterns.  
- **Resample**: aggregates higher-frequency data into lower-frequency intervals; aligns series with different granularities.  
- **Cumulative Sum**: converts flow data into stock format (inverse of difference).  

---

### Backtest
The core system component, allowing users to run backtests by submitting parameters via **POST** requests:
- **Dataset identifier**  
- **Model type**: supported models include `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`  
- **Hyperparameter set**: dictionary mapping each parameter to a list of values to be tested  
- **Tuning frequency**: integer defining how often Grid Search runs to find optimal hyperparameters  
- **Window type**: rolling or expanding (defines validation approach)  
- **Window size** (if rolling)  
- **Forecast horizon**: number of steps ahead to predict  
- **Standardize**: apply `StandardScaler` at each iteration (similar to Z-score)  
- **Parallelize**: run backtests of each model in parallel  
- **Error metrics**: RMSE, MAE, MAPE, etc.  

