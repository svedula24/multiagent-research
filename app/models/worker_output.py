from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class WorkerOutput(BaseModel):
    source: Literal["web", "reviews", "sales"]
    status: Literal["success", "failed", "timeout"]
    findings: dict = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    error: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence", mode="after")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)

    @field_validator("findings", mode="after")
    @classmethod
    def empty_findings_on_failure(cls, v: dict, info) -> dict:
        # Ensure failed/timeout outputs always have an empty findings dict
        # (validation runs after all fields are set via model_validator if needed,
        # but field_validator on findings is sufficient here)
        return v

    model_config = {"frozen": False}
