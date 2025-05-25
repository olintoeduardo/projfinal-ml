from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.responses import JSONResponse

UPLOAD_DIR = Path("uploads") # onde os arquivos são salvos
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter(prefix="/datasets", tags=["datasets"])

# armazenamento em memória (metadata apenas)
_DATA: Dict[int, dict] = {}
_COUNTER = 0


@router.post("", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Recebe CSV ou Excel e salva *o próprio arquivo* em uploads/.
    """
    if not file.filename.endswith((".csv", ".xls", ".xlsx")):
        raise HTTPException(415, "Só aceito CSV ou Excel")

    raw_bytes = await file.read()
    saved_path = UPLOAD_DIR / f"{int(datetime.utcnow().timestamp())}_{file.filename}"
    saved_path.write_bytes(raw_bytes)

    # Lê só para pegar metadados (colunas, rows)
    if file.filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(saved_path)
    else:
        df = pd.read_csv(saved_path)

    global _COUNTER
    _COUNTER += 1
    _DATA[_COUNTER] = {
        "path": str(saved_path),
        "name": file.filename,
        "columns": list(df.columns),
        "rows": len(df),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    return {"id": _COUNTER, **_DATA[_COUNTER]}


@router.get("/{ds_id}")
def dataset_info(ds_id: int):
    if ds_id not in _DATA:
        raise HTTPException(404, "ID não encontrado")
    return _DATA[ds_id]


@router.get("/{ds_id}/sample")
def dataset_sample(ds_id: int, n: int = 5):
    if ds_id not in _DATA:
        raise HTTPException(404, "ID não encontrado")

    path = _DATA[ds_id]["path"]
    if path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, nrows=n)
    else:
        df = pd.read_csv(path, nrows=n)

    return JSONResponse(df.to_dict(orient="records"))