# app/schemas/transform.py
from typing import Union
from pydantic import BaseModel
from typing import Union

class PctChangeReq(BaseModel):
    operation: str
    column: str
    periods: int

class DiffReq(BaseModel):
    operation: str
    column: str
    periods: int

class RollingMeanReq(BaseModel):
    operation: str
    column: str
    window: int

class ResampleReq(BaseModel):
    operation: str
    column: str
    freq: str
    how: str = "mean"

class CumsumReq(BaseModel):
    operation: str
    column: str

TransformRequest = Union[
    PctChangeReq, DiffReq, RollingMeanReq, ResampleReq, CumsumReq
]