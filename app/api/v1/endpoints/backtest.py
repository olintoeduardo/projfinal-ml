from fastapi import APIRouter, HTTPException

from app.schemas.backtest import BacktestRequest, BacktestResult
from app.services.backtest.backtest_service import BacktestService

router = APIRouter(
    prefix="/backtest",
    tags=["backtest"],
)

# Instância única do serviço
service = BacktestService()

@router.post("",
             response_model=BacktestResult,
             status_code=201)
def create_backtest(request: BacktestRequest):
    """
    Endpoint para iniciar um backtest.
    Recebe parâmetros via BacktestRequest e retorna o resultado completo.
    """
    try:
        result = service.run_backtest(
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            hyperparams=request.hyperparams,
            window_type=request.window_type,
            window_size=request.window_size,
            horizon=request.horizon,
            metrics=request.metrics,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
