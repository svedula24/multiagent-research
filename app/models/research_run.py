import re

from pydantic import BaseModel, Field, field_validator


class ResearchRunRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=1000, description="Research goal")
    competitors: list[str] = Field(
        ..., min_length=1, max_length=10, description="List of competitor names"
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Strip leading/trailing whitespace; reject obvious prompt injection attempts
        v = v.strip()
        if re.search(r"(ignore previous|system prompt|</s>|<\|im_start\|>)", v, re.IGNORECASE):
            raise ValueError("Query contains disallowed content")
        return v

    @field_validator("competitors", mode="before")
    @classmethod
    def parse_competitors(cls, v):
        # Accept either a list or a comma-separated string from the Streamlit form
        if isinstance(v, str):
            v = [c.strip() for c in v.split(",") if c.strip()]
        return v

    @field_validator("competitors", mode="after")
    @classmethod
    def sanitize_competitors(cls, v: list[str]) -> list[str]:
        cleaned = []
        for name in v:
            name = name.strip()
            # Allow only alphanumeric, spaces, hyphens, dots, ampersands
            if not re.match(r"^[\w\s\-\.&]+$", name):
                raise ValueError(f"Competitor name contains invalid characters: {name!r}")
            cleaned.append(name[:100])  # cap length
        return cleaned


class ResearchRunResponse(BaseModel):
    run_id: int
    status: str = "running"
