from pathlib import Path
import datetime as dt

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from app.api.v1.endpoints.datasets import _build_path, _read_df  # helpers

UPLOAD_DIR = Path("uploads")

router = APIRouter(
    prefix="/transform",
    tags=["transformations"],
)

# ───────── helpers atualizados ────────────────────────────────────────
def _finish_samefile(df_out: pd.DataFrame, base_path: Path):
    preview = df_out.head(5).reset_index()

    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
        elif isinstance(preview[col].iloc[0], (pd.Timestamp, dt.datetime)):
            preview[col] = preview[col].map(lambda x: x.isoformat())

    payload = {
        "file_saved": str(base_path.relative_to(UPLOAD_DIR)),
        "rows": len(df_out),
        "columns": list(df_out.columns),
        "preview": preview.to_dict(orient="records"),
    }
    return JSONResponse(jsonable_encoder(payload), status_code=201)


def _read_full_df(dataset_id: str) -> tuple[pd.DataFrame, Path]:
    path = _build_path(dataset_id)
    df = _read_df(path)
    return df, path


# ───────── endpoints ────────────────────────────────────────────
@router.post("/pct_change", status_code=201)
def pct_change(
    dataset_id: str,
    column: str = Query(...),
    periods: int = Query(..., ge=1),
):
    df, path = _read_full_df(dataset_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")

    new_col = f"{column}_pct_change_{periods}"
    df[new_col] = df[column].pct_change(periods)
    df = df.dropna(subset=[new_col])
    df.to_csv(path, index=True, sep=",")
    return _finish_samefile(df, path)


@router.post("/diff", status_code=201)
def diff(
    dataset_id: str,
    column: str = Query(...),
    periods: int = Query(..., ge=1),
):
    df, path = _read_full_df(dataset_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")

    new_col = f"{column}_diff_{periods}"
    df[new_col] = df[column].diff(periods)
    df = df.dropna(subset=[new_col])
    df.to_csv(path, index=True, sep=",")
    return _finish_samefile(df, path)


@router.post("/rolling_mean", status_code=201)
def rolling_mean(
    dataset_id: str,
    column: str = Query(...),
    window: int = Query(..., ge=1),
):
    df, path = _read_full_df(dataset_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")

    new_col = f"{column}_rollmean_{window}"
    df[new_col] = df[column].rolling(window).mean()
    df = df.dropna(subset=[new_col])
    df.to_csv(path, index=True, sep=",")
    return _finish_samefile(df, path)


@router.post("/resample", status_code=201)
def resample(
    dataset_id: str,
    column: str = Query(...),
    freq: str = Query(..., examples=["M", "Q", "A"]),
    how: str = Query("mean", examples=["mean", "sum", "ffill", "bfill"]),
):
    df, path = _read_full_df(dataset_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise HTTPException(400, "Index must be a datetime for resampling")

    s = df[column]
    if how in {"ffill", "bfill"}:
        resampled = s.resample(freq).ffill() if how == "ffill" else s.resample(freq).bfill()
    else:
        resampled = s.resample(freq).agg(how)

    new_col = f"{column}_resample_{freq}_{how}"
    new_df = resampled.to_frame(new_col).dropna()
    df_resampled = new_df  # sobrescreve com somente as datas válidas
    df_resampled.to_csv(path, index=True, sep=",")
    return _finish_samefile(df_resampled, path)


@router.post("/cumsum", status_code=201)
def cumsum(
    dataset_id: str,
    column: str = Query(...),
):
    df, path = _read_full_df(dataset_id)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")

    new_col = f"{column}_cumsum"
    df[new_col] = df[column].cumsum()
    df = df.dropna(subset=[new_col])
    df.to_csv(path, index=True, sep=",")
    return _finish_samefile(df, path)