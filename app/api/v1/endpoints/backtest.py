from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends

from app.services.backtest.backtest_service import BacktestService
from app.schemas.backtest import BacktestResult
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
import json

def get_hyperparams_dict(hyperparameters: List[str] = Query(...)):
    return map(json.loads, hyperparameters)

router = APIRouter(prefix="/backtests", tags=["backtests"])
service = BacktestService()

@router.post("", response_model=BacktestResult, status_code=201)
def create_backtest(
    dataset_id: str = Query(..., description="ID do dataset (timestamp)"),
    target_column: str = Query(..., description="Nome da coluna alvo"),
    feature_columns: List[str] = Query(
        ..., description="colunas independentes)"
    ),
    model_type: str = Query(..., description="Tipo do modelo (ridge, lasso, random_forest)"),
    hyperparams: list = Depends(get_hyperparams_dict),
    window_type: str = Query("rolling", description="rolling ou expanding"),
    window_size: Optional[int] = Query(
        None, ge=1, description="Tamanho da janela"
    ),
    initial_window: Optional[int] = Query(
        0, ge=0, description="Tamanho da janela inicial"
    ),
    horizon: int = Query(..., ge=1, description="Passos à frente para previsão"),
    metrics: List[str] = Query(
        ["rmse", "mae"], description="Métricas (ex: ?metrics=rmse&metrics=mae)"
    ),
):
    try:
        hp_dict = {}
        for value in hyperparams:
            hp_dict.update(value)
            
        result = service.run_backtest(
            dataset_id=dataset_id,
            target_column=target_column,
            feature_columns=feature_columns,
            model_type=model_type,
            hyperparams=hp_dict,
            window_type=window_type,
            window_size=window_size,
            initial_window=initial_window,
            horizon=horizon,
            metrics=metrics,
        )
        return JSONResponse(jsonable_encoder(result), status_code=201)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))