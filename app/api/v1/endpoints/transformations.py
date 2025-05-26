from pathlib import Path
import datetime as dt

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from app.api.v1.endpoints.datasets import _build_path, _read_df  # helpers

UPLOAD_DIR = Path("uploads")

# agora o dataset_id faz parte do path
router = APIRouter(
    prefix="/transform",
    tags=["transformations"],
)


# ───────── helpers comuns ────────────────────────────────────────
def _save(df: pd.DataFrame, base_path: Path, suffix: str) -> Path:
    new_name = f"{base_path.stem}__{suffix}.csv"
    out_path = base_path.with_name(new_name)
    df.to_csv(out_path, index=True, sep=";")
    return out_path


def _finish(df_out: pd.DataFrame, out_path: Path):
    preview = df_out.head(5).reset_index()

    # convert Timestamps → string
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
        elif isinstance(preview[col].iloc[0], (pd.Timestamp, dt.datetime)):
            preview[col] = preview[col].map(lambda x: x.isoformat())

    payload = {
        "new_file": str(out_path.relative_to(UPLOAD_DIR)),
        "rows": len(df_out),
        "columns": list(df_out.columns),
        "preview": preview.to_dict(orient="records"),
    }
    return JSONResponse(jsonable_encoder(payload), status_code=201)


def _base_series(dataset_id: str, column: str) -> tuple[pd.Series, Path]:
    path = _build_path(dataset_id)
    df = _read_df(path).dropna()
    if column not in df.columns:
        raise HTTPException(400, f"Coluna {column} não encontrada")
    return df[column], path


# ───────── endpoints ────────────────────────────────────────────
@router.post("/pct_change", status_code=201)
def pct_change(
    dataset_id: str,
    column: str = Query(..., description="Nome da coluna"),
    periods: int = Query(..., ge=1, description="Nº períodos"),
):
    s, base_path = _base_series(dataset_id, column)
    df_out = s.pct_change(periods).to_frame(f"{column}_pct_change_{periods}").dropna().reset_index()
    return _finish(df_out, _save(df_out, base_path, f"pct_change_{periods}"))


@router.post("/diff", status_code=201)
def diff(
    dataset_id: str,
    column: str = Query(...),
    periods: int = Query(..., ge=1),
):
    s, base_path = _base_series(dataset_id, column)
    df_out = s.diff(periods).to_frame(f"{column}_diff_{periods}").dropna().reset_index()
    return _finish(df_out, _save(df_out, base_path, f"diff_{periods}"))


@router.post("/rolling_mean", status_code=201)
def rolling_mean(
    dataset_id: str,
    column: str = Query(...),
    window: int = Query(..., ge=1),
):
    s, base_path = _base_series(dataset_id, column)
    df_out = s.rolling(window).mean().to_frame(f"{column}_rollmean_{window}").dropna().reset_index()
    return _finish(df_out, _save(df_out, base_path, f"rollmean_{window}"))


@router.post("/resample", status_code=201)
def resample(
    dataset_id: str,
    column: str = Query(...),
    freq: str = Query(..., examples=["M", "Q", "A"]),
    how: str = Query("mean", examples=["mean", "sum", "ffill", "bfill"]),
):
    s, base_path = _base_series(dataset_id, column)
    if how in {"ffill", "bfill"}:
        res = s.resample(freq).ffill() if how == "ffill" else s.resample(freq).bfill()
    else:
        res = s.resample(freq).agg(how)
    df_out = res.to_frame(f"{column}_resample_{freq}_{how}").dropna().reset_index()
    return _finish(df_out, _save(df_out, base_path, f"resample_{freq}_{how}"))


@router.post("/cumsum", status_code=201)
def cumsum(
    dataset_id: str,
    column: str = Query(...),
):
    s, base_path = _base_series(dataset_id, column)
    df_out = s.cumsum().to_frame(f"{column}_cumsum").dropna().reset_index()
    return _finish(df_out, _save(df_out, base_path, "cumsum"))