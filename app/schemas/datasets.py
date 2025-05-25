from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

class DatasetInfo(BaseModel):
    """Metadados básicos de um dataset carregado na sessão."""

    id: int
    name: str = Field(..., description="Nome original do arquivo")
    columns: List[str]
    rows: int
    uploaded_at: datetime