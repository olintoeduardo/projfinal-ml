from fastapi import APIRouter

from app.api.v1.endpoints import datasets

api_router = APIRouter(prefix="/api/v1")

# cada include mantém seu próprio prefixo/tags
api_router.include_router(datasets.router)

# api_router.include_router(models.router)
# api_router.include_router(backtests.router)