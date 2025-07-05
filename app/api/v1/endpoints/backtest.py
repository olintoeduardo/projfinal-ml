# app/api/v1/endpoints/backtests.py
from typing import Dict, List, Tuple, Optional
from fastapi import APIRouter, Query, Depends, HTTPException
from starlette.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json

from app.services.backtest.backtest_service import BacktestService
from app.schemas.backtest import BacktestResult

router = APIRouter(prefix="/backtests", tags=["backtests"])
service = BacktestService()


# -------- utilitário: converte lista de JSONs em dicts ----------------
def _parse_hyperparams(params: List[str] = Query(...)) -> List[Dict]:
    """Cada item deve vir como JSON: {"alpha":[0.1,1]}"""
    try:
        return list(map(json.loads, params))
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Bad hyperparam JSON: {e}")


@router.post(
    "",
    response_model=Dict[str, BacktestResult],   # <- vários resultados
    status_code=201,
)
def create_backtests(
    # --- dados principais ---
    dataset_id: str = Query(..., description="ID do dataset"),
    target_column: str = Query(...),
    feature_columns: List[str] = Query(...),

    # --- suporte a múltiplos modelos ---
    model_types: List[str] = Query(
        ...,
        description="Repita ?model_types=ridge&model_types=lasso …",
    ),
    hyperparams: List[Dict] = Depends(_parse_hyperparams),
    tuning_frequency: int = Query(
        10,
        ge=1,
        description="Reexecutar GridSearch a cada N observações",
    ),
    # --- config de janela / métrica / horizonte ---
    window_type: str = Query("rolling", regex="^(rolling|expanding)$"),
    window_size: Optional[int] = Query(None, ge=1),
    initial_window: Optional[int] = Query(0, ge=0),
    horizon: int = Query(1, ge=0, description="0 = nowcast"),
    metrics: List[str] = Query(["rmse", "mae"]),

    # --- novos flags ---
    execution_mode: str = Query(
        "sequential", regex="^(sequential|parallel)$",
        description="Como executar vários modelos",
    ),
    is_nowcast: bool = Query(
        False,
        description="True força horizon=0; se False usa 'horizon' informado",
    ),
):
    # ------------------------ validações -----------------------------
    if len(model_types) != len(hyperparams):
        raise HTTPException(
            400, "model_types e hyperparams devem ter o mesmo comprimento"
        )

    # sobrepõe horizon se user escolheu flag nowcast
    horizon = 0 if is_nowcast else horizon

    # monta lista [(modelo, dict_hps), …]
    model_specs: List[Tuple[str, Dict]] = [
        (m, hp) for m, hp in zip(model_types, hyperparams)
    ]

    try:
        if execution_mode == "parallel":
            results = service.run_backtests_parallel(
                dataset_id=dataset_id,
                target_column=target_column,
                feature_columns=feature_columns,
                model_specs=model_specs,
                tuning_frequency=tuning_frequency,
                window_type=window_type,
                window_size=window_size,
                initial_window=initial_window,
                horizon=horizon,
                metrics=metrics,
            )
        else:  # sequential
            results = service.run_backtests_sequential(
                dataset_id=dataset_id,
                target_column=target_column,
                feature_columns=feature_columns,
                model_specs=model_specs,
                tuning_frequency=tuning_frequency,
                window_type=window_type,
                window_size=window_size,
                initial_window=initial_window,
                horizon=horizon,
                metrics=metrics,
            )

        return JSONResponse(jsonable_encoder(results), status_code=201)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))