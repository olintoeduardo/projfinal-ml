from fastapi import APIRouter

from app.api.v1.endpoints import datasets, transformations, backtest

api_router = APIRouter()

# cada include mantém seu próprio prefixo/tags
api_router.include_router(datasets.router)
api_router.include_router(transformations.router)
api_router.include_router(backtest.router)

# api_router.include_router(models.router)
# api_router.include_router(backtests.router)