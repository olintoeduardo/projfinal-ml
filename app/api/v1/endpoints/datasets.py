from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.responses import JSONResponse

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _build_path(ds_id: str) -> Path:
    """Encontra o arquivo que começa com <ds_id>_ dentro de uploads/."""
    candidates = list(UPLOAD_DIR.glob(f"{ds_id}.csv"))
    if not candidates:
        raise HTTPException(404, detail="Dataset não encontrado")
    # Se houver duplicados, pega o primeiro (esperado 1-para-1)
    return candidates[0]


def _read_df(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows, sep=";", index_col=0)


def _metadata(path: Path) -> dict:
    df_head = _read_df(path, nrows=0)   # lê só header → colunas
    rows = sum(1 for _ in open(path, "rb")) - 1 if path.suffix == ".csv" else None
    # Para Excel não há leitura rápida de linhas sem abrir toda planilha,
    # então usa len(df) (custa mais, mas ok para MVP).
    if rows is None:
        rows = len(_read_df(path))
    return {
        "id": path.stem.split("_")[0],
        "name": path.name.split("_", 1)[1],
        "columns": list(df_head.columns),
        "rows": rows,
        "uploaded_at": datetime.utcfromtimestamp(int(path.stem.split("_")[0])).isoformat()
    }


@router.post("", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV/XLSX e salva em uploads/ usando timestamp como id."""
    if not file.filename.endswith((".csv", ".xls", ".xlsx")):
        raise HTTPException(415, "Só aceito CSV ou Excel")

    ds_id = 0
    saved_path = UPLOAD_DIR / f"{ds_id}_{file.filename}"
    saved_path.write_bytes(await file.read())

    return _metadata(saved_path)


@router.get("/info")
def dataset_info(ds_id: str):
    """Retorna metadados do dataset identificado pelo timestamp."""
    path = _build_path(ds_id)
    return _metadata(path)


@router.get("/sample")
def dataset_sample(ds_id: str, n: int = 5):
    """Devolve as *n* primeiras linhas em JSON."""
    path = _build_path(ds_id)
    df = _read_df(path, nrows=n)
    return JSONResponse(df.to_dict(orient="records"))