# app/services/backtest_service.py
from typing import Optional, Dict, List
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.schemas.backtest import BacktestResult, MetricResult, PredictionPoint
from app.api.v1.endpoints.datasets import _build_path, _read_df

class BacktestService:
    def run_backtest(
        self,
        dataset_id: str,
        target_column: str,
        feature_columns: List[str],
        model_type: str,
        hyperparams: Dict[str, float],
        window_type: str,
        window_size: Optional[int],
        initial_window: Optional[int],
        horizon: int,
        metrics: List[str],
    ) -> BacktestResult:
        # Gera um ID único para este backtest
        bt_id = uuid.uuid4().hex
        np.random.seed(0)

        # Carrega série
        path: Path = _build_path(dataset_id)
        df: pd.DataFrame = _read_df(path).dropna()
        df.index = pd.to_datetime(df.index)

        # Verifica colunas
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found")

        # Define X e y
        y = df[target_column]
        if feature_columns:
            X_df = df[feature_columns]
        else:
            # Se não houver features, usa índice como única feature
            X_df = pd.DataFrame({"__time_idx": np.arange(len(y))}, index=y.index)

        dates = y.index
        y_true: List[float] = []
        y_pred: List[float] = []
        date_list: List[str] = []

        win_type = window_type
        win_size = window_size or len(y)

        # Loop de backtest
        for i in range(initial_window,len(y) - horizon):
            if win_type == "rolling":
                start = max(0, i)
                end_train = start + win_size
                y_train = y.iloc[start:end_train]
                X_train = X_df.iloc[start:end_train]
                pred_idx = end_train + horizon - 1
            else:  # expanding
                y_train = y.iloc[: i + 1]
                X_train = X_df.iloc[: i + 1]
                pred_idx = i + horizon

            if pred_idx >= len(y):
                break

            # Inicializa e treina o modelo
            model = self._init_model(model_type, hyperparams)
            model.fit(X_train.values, y_train.values)

            # Previsão
            X_pred = X_df.iloc[[pred_idx]].values
            pred_val = float(model.predict(X_pred)[0])
            true_val = float(y.iloc[pred_idx])

            y_pred.append(pred_val)
            y_true.append(true_val)
            date_list.append(dates[pred_idx].isoformat())

        # Cálculo de métricas
        arr_true = np.array(y_true)
        arr_pred = np.array(y_pred)
        metric_results: List[MetricResult] = []
        for m in metrics:
            if m == "rmse":
                val = float(np.sqrt(mean_squared_error(arr_true, arr_pred)))
            elif m == "mae":
                val = float(mean_absolute_error(arr_true, arr_pred))

            else:
                continue
            metric_results.append(MetricResult(name=m, value=val))

        # Monta série de previsões
        prediction_points = [
            PredictionPoint(date=date_list[i], y_true=y_true[i], y_pred=y_pred[i])
            for i in range(len(y_true))
        ]

        return BacktestResult(
            id=bt_id,
            status="completed",
            metrics=metric_results,
            predictions=prediction_points,
        )


    def _init_model(self, model_type: str, hyperparams: Dict[str, float]):
        """Factory simples para selecionar e configurar o modelo."""
        if model_type == "ridge":
            return Ridge()
        if model_type == "lasso":
            return Lasso()
        if model_type == "random_forest":
            return RandomForestRegressor()
        raise ValueError(f"Modelo não suportado: {model_type}")