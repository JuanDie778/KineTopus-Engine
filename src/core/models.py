from typing import List, Optional, Any
from pydantic import BaseModel, Field

class ColumnMetadata(BaseModel):
    """Metadatos de una columna específica."""
    name: str
    inferred_type: str  # 'numeric', 'categorical', 'datetime', 'text', 'boolean'
    missing_percentage: float
    unique_values_count: int
    sample_values: List[Any] = Field(default_factory=list)

class DatasetMetadata(BaseModel):
    """Metadatos globales del dataset."""
    file_name: str
    file_size_mb: float
    total_rows: int
    total_columns: int
    columns: List[ColumnMetadata] = Field(default_factory=list)
    technical_summary: Optional[str] = None
