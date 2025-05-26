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
    """
    Serviço único para executar backtests
    sem acoplamento direto aos endpoints.
    """

    def run_backtest(
        self,
        dataset_id: str,
        model_type: str,
        hyperparams: Dict[str, float],
        window_type: str,
        window_size: Optional[int],
        horizon: int,
        metrics: List[str],
    ) -> BacktestResult:
        # Gera um ID único para este backtest
        bt_id = uuid.uuid4().hex
        np.random.seed(0)

        # Carrega série
        path: Path = _build_path(dataset_id)
        df: pd.DataFrame = _read_df(path).dropna()
        df.index = pd.to_datetime(df.iloc[:, 0])
        y = df.iloc[:, 1]
        dates = y.index

        # Preparação de listas de resultados
        y_true, y_pred, date_list = [], [], []

        # Loop de janela
        for i in range(len(y) - horizon):
            if window_type == "rolling":
                start = max(0, i)
                train = y.iloc[start : start + window_size]
                pred_idx = start + window_size + horizon - 1
            else:  # expanding
                train = y.iloc[: i + 1]
                pred_idx = i + horizon

            if pred_idx >= len(y):
                break

            # Inicializa e treina o modelo
            model = self._init_model(model_type, hyperparams)
            X_train = np.arange(len(train)).reshape(-1, 1)
            model.fit(X_train, train.values)

            # Previsão
            X_pred = np.array([[len(train) + horizon - 1]])
            pred_val = float(model.predict(X_pred)[0])
            true_val = float(y.iloc[pred_idx])

            date_list.append(dates[pred_idx].isoformat())
            y_true.append(true_val)
            y_pred.append(pred_val)

        # Cálculo de métricas
        arr_true = np.array(y_true)
        arr_pred = np.array(y_pred)
        metric_results: List[MetricResult] = []
        for m in metrics:
            if m == "rmse":
                val = float(np.sqrt(mean_squared_error(arr_true, arr_pred)))
            elif m == "mae":
                val = float(mean_absolute_error(arr_true, arr_pred))
            elif m == "mape":
                val = float(np.mean(np.abs((arr_true - arr_pred) / arr_true))) * 100
            else:
                continue
            metric_results.append(MetricResult(name=m, value=val))

        # Ponto a ponto
        prediction_points = [
            PredictionPoint(date=date_list[i], y_true=y_true[i], y_pred=y_pred[i])
            for i in range(len(y_true))
        ]

        # Retorna o resultado completo
        return BacktestResult(
            id=bt_id,
            status="completed",
            metrics=metric_results,
            predictions=prediction_points,
        )

    def _init_model(self, model_type: str, hyperparams: Dict[str, float]):
        """Factory simples para selecionar e configurar o modelo."""
        if model_type == "ridge":
            return Ridge(**hyperparams)
        if model_type == "lasso":
            return Lasso(**hyperparams)
        if model_type == "random_forest":
            return RandomForestRegressor(**hyperparams)
        raise ValueError(f"Modelo não suportado: {model_type}")