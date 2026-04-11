from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DraftRecommendation(BaseModel):
    run_id: int
    summary: str = Field(..., description="2-3 sentence executive summary")
    findings_by_source: dict = Field(
        default_factory=dict,
        description="Key findings keyed by source: web, reviews, sales",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Synthesis confidence score")
    status: Literal["pending_approval", "approved", "rejected", "failed"] = "pending_approval"
    rejection_count: int = Field(default=0, ge=0, le=3)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence", mode="after")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)


class FinalRecommendation(BaseModel):
    run_id: int
    content: str = Field(..., description="Full approved recommendation text")
    confidence: float = Field(..., ge=0.0, le=1.0)
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence", mode="after")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)


class RejectionRequest(BaseModel):
    feedback: Optional[str] = Field(
        None, max_length=2000, description="Optional PM feedback for re-synthesis"
    )
