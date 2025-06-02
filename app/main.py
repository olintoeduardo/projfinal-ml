from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


from app.api.v1.api import api_router

app = FastAPI(title="Time-Series Backtest API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app)
