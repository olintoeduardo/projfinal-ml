from typing import Optional, Dict, List
from pydantic import BaseModel

class BacktestRequest(BaseModel):
    dataset_id: str
    model_type: str               # ex: "ridge", "lasso", "random_forest"
    hyperparams: Dict[str, List[float]] = {}
    window_type: str = "rolling"  # "expanding" ou "rolling"
    window_size: Optional[int] = None  # obrigatório se window_type == "rolling"
    horizon: int
    metrics: List[str] = ["rmse", "mae", "mape"]


class BacktestStatus(BaseModel):
    id: str
    status: str                   # "running", "completed", "failed"


class MetricResult(BaseModel):
    name: str                     # "rmse", "mae", "mape"
    value: float


class PredictionPoint(BaseModel):
    date: str                     # ISO format
    y_true: float
    y_pred: float
    hyperparams: Dict[str, float]  # valores escalares



class BacktestResult(BaseModel):
    id: str
    status: str                   # deve ser "completed" ou "failed"
    metrics: List[MetricResult]
    predictions: List
