from typing import Dict, List, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.services.backtest.backtest_service import BacktestService
from app.schemas.backtest import BacktestResult

router = APIRouter(prefix="/backtests", tags=["backtests"])
service = BacktestService()


class ModelSpec(BaseModel):
    model_type: str
    hyperparams: Dict


class BacktestRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    model_specs: List[ModelSpec]
    tuning_frequency: int = 10
    window_type: str = "rolling"  # "rolling" or "expanding"
    window_size: int | None = None
    initial_window: int = 0
    horizon: int = 1
    metrics: List[str] = ["rmse", "mae"]
    execution_mode: str = "sequential"  # "sequential" or "parallel"
    is_nowcast: bool = False
    standardize: bool = True


@router.post(
    "",
    response_model=Dict[str, BacktestResult],
    status_code=201,
)
def create_backtests(req: BacktestRequest):
    if req.window_type not in {"rolling", "expanding"}:
        raise HTTPException(status_code=400, detail="Invalid window_type")

    if req.execution_mode not in {"sequential", "parallel"}:
        raise HTTPException(status_code=400, detail="Invalid execution_mode")

    if len(req.model_specs) == 0:
        raise HTTPException(status_code=400, detail="No models provided")

    model_specs: List[Tuple[str, Dict]] = [
        (m.model_type, m.hyperparams) for m in req.model_specs
    ]

    try:
        horizon = 0 if req.is_nowcast else req.horizon

        if req.execution_mode == "parallel":
            results = service.run_backtests_parallel(
                dataset_id=req.dataset_id,
                target_column=req.target_column,
                feature_columns=req.feature_columns,
                model_specs=model_specs,
                tuning_frequency=req.tuning_frequency,
                window_type=req.window_type,
                window_size=req.window_size,
                initial_window=req.initial_window,
                horizon=horizon,
                metrics=req.metrics,
                standardize=req.standardize
            )
        else:
            results = service.run_backtests_sequential(
                dataset_id=req.dataset_id,
                target_column=req.target_column,
                feature_columns=req.feature_columns,
                model_specs=model_specs,
                tuning_frequency=req.tuning_frequency,
                window_type=req.window_type,
                window_size=req.window_size,
                initial_window=req.initial_window,
                horizon=horizon,
                metrics=req.metrics,
                standardize=req.standardize
            )

        return JSONResponse(jsonable_encoder(results), status_code=201)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))