from typing import Optional, Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")

from app.schemas.backtest import BacktestResult, MetricResult, PredictionPoint
from app.api.v1.endpoints.datasets import _build_path, _read_df

class BacktestService:
    def grid_search_tscv(self,
        X: pd.DataFrame,
        y: pd.Series,
        model_class: Any,
        param_grid: Dict[str, list],
        n_splits: int = 5,
        scoring: str | None = "neg_mean_squared_error",
        n_jobs: int = 1,
        verbose: int = 0,
        # **model_kwargs: Any,
    ) -> GridSearchCV:

        tscv = TimeSeriesSplit(n_splits=n_splits)

        base_model = model_class(random_state=0)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
            error_score="raise",
        )

        grid.fit(X, y)

        return grid

    def run_single_backtest(
        self,
        dataset_id: str,
        target_column: str,
        feature_columns: List[str],
        model_type: str,
        hyperparams: Dict[str, float],
        tuning_frequency: int,
        window_type: str,
        window_size: Optional[int] = None,
        initial_window: Optional[int] = 0,
        horizon: int = 1,
        metrics: List[str] = ["rmse"],
        n_jobs: int = -1,
        lags: int = None,
        is_nowcast : bool = False,
    ) -> BacktestResult:
        
        bt_id = uuid.uuid4().hex
        np.random.seed(0)

        path: Path = _build_path(dataset_id)
        df: pd.DataFrame = _read_df(path).dropna()
        df.index = pd.to_datetime(df.index)

        # Verifica colunas
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found")

        # IMPORTANT: Now cast uses same date, forecast uses one after
        if not is_nowcast:
            df[target_column] = df[target_column].shift(-horizon)
            df.dropna()

        if lags:
            for lag in range(lags):
                if not is_nowcast:
                    lag += 1
                    
                df[f"lag_{lag}"] = df[target_column].shift(-lag)

        df = df.dropna()
        y = df[target_column]
        X_df = df[feature_columns]

        
        y_true: List[float] = []
        y_pred: List[float] = []
        date_list: List[str] = []
        best_params = {}

        for i in range(initial_window,len(y) - horizon):
            if window_type == "rolling":
                start = max(0, i)
                end_train = start + window_size
                y_train = y.iloc[start:end_train]
                X_train = X_df.iloc[start:end_train]
                pred_idx = end_train + horizon - 1
            else:
                y_train = y.iloc[: i]
                
                X_train = X_df.iloc[: i]
                pred_idx = i + horizon - 1

            if pred_idx >= len(y):
                break

            if i % tuning_frequency == 0:
                grid_search = self.grid_search_tscv(X=X_train,
                                                    y=y_train,
                                                    model_class=self.get_model_class(model_type), 
                                                    param_grid=hyperparams,
                                                    n_jobs=n_jobs
                                                    )
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
            model = self.get_model_instance(model_type, best_params)
            model.fit(X_train.values, y_train.values)

            X_pred = X_df.iloc[[pred_idx]].values
            pred_val = float(model.predict(X_pred)[0])
            true_val = float(y.iloc[pred_idx])

            y_pred.append(pred_val)
            y_true.append(true_val)
            date_list.append(str(X_df.index[pred_idx]))

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

        prediction_points = [
            PredictionPoint(date=date_list[i], y_true=y_true[i], y_pred=y_pred[i], hyperparams=best_params)
            for i in range(len(y_true))
        ]

        return model_type, BacktestResult(
            id=bt_id,
            status="completed",
            metrics=metric_results,
            predictions=prediction_points,
        )
    
    def run_backtests_sequential(
        self,
        dataset_id: str,
        target_column: str,
        feature_columns: List[str],
        model_specs: List[Tuple[str, Dict[str, Any]]],  # [(model_type, hyperparams), ...]
        tuning_frequency: int,
        window_type: str,
        window_size: Optional[int] = None,
        initial_window: int = 0,
        horizon: int = 1,
        metrics: List[str] = ["rmse"],
    ):
        results: Dict[str, BacktestResult] = {}
        t0 = time.perf_counter()
        for model_type, hyperparams in model_specs:
            model_type, backtest_result = self.run_single_backtest(
                dataset_id=dataset_id,
                target_column=target_column,
                feature_columns=feature_columns,
                model_type=model_type,
                hyperparams=hyperparams,
                tuning_frequency=tuning_frequency,
                window_type=window_type,
                window_size=window_size,
                initial_window=initial_window,
                horizon=horizon,
                metrics=metrics,
                n_jobs=-1
            )
            results[model_type] = backtest_result
        
        elapsed = time.perf_counter() - t0
        results['time'] = elapsed
        return results

    def run_backtests_parallel(
        self,
        dataset_id: str,
        target_column: str,
        feature_columns: List[str],
        model_specs: List[Tuple[str, Dict[str, Any]]],  # [(model_type, hyperparams), ...]
        tuning_frequency: int,
        window_type: str,
        window_size: Optional[int] = None,
        initial_window: int = 0,
        horizon: int = 1,
        metrics: List[str] = ["rmse"],
        max_workers: Optional[int] = None,
    ) -> Dict[str, BacktestResult]:
        results: Dict[str, BacktestResult] = {}
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.run_single_backtest,
                    dataset_id,
                    target_column,
                    feature_columns,
                    model_type,
                    hyperparams,
                    tuning_frequency,
                    window_type,
                    window_size,
                    initial_window,
                    horizon,
                    metrics,
                    n_jobs=1
                )
                for model_type, hyperparams in model_specs
            ]

            for fut in as_completed(futures):
                model_type, backtest_result = fut.result()
                results[model_type] = backtest_result
        
        elapsed = time.perf_counter() - t0
        results['time'] = elapsed
        return results

    def get_model_instance(self, model_type: str, hyperparams: Dict[str, float]):
        """Factory simples para selecionar e configurar o modelo."""
        if model_type == "ridge":
            return Ridge(**hyperparams)
        if model_type == "lasso":
            return Lasso(**hyperparams)
        if model_type == "random_forest":
            return RandomForestRegressor(**hyperparams)
        raise ValueError(f"Modelo não suportado: {model_type}")

    def get_model_class(self, model_type):
        if model_type == "ridge":
            return Ridge
        if model_type == "lasso":
            return Lasso
        if model_type == "random_forest":
            return RandomForestRegressor
        raise ValueError(f"Modelo não suportado: {model_type}")
        
if __name__ == "__main__":
    service = BacktestService()
    model_specs = [
        ("ridge", {"alpha": [0.1, 1.0, 10.0]}),        # grid de alphas
        ("lasso", {"alpha": [0.01, 0.1, 1.0]}),
        # ("random_forest", {"n_estimators": [100, 300]}),
    ]
    results_seq = service.run_backtests_sequential(
        dataset_id="0",
        target_column="X1",
        feature_columns=["X2"],
        model_specs=model_specs,
        tuning_frequency=10,
        window_type="expanding",
        initial_window=5,
        horizon=1,
        metrics=["rmse"]
    )

    results_para = service.run_backtests_parallel(
        dataset_id="0",
        target_column="X1",
        feature_columns=["X2"],
        model_specs=model_specs,
        tuning_frequency=10,
        window_type="expanding",
        initial_window=5,
        horizon=1,
        metrics=["rmse"]
    )

    results_para

