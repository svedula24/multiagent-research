"""
Shared pytest fixtures for unit tests.
All external dependencies (DB, LLM, APIs) are mocked here.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Sample WorkerOutput fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def web_output_success():
    from app.models.worker_output import WorkerOutput
    return WorkerOutput(
        source="web",
        status="success",
        findings={
            "results": [
                {"competitor": "Amplitude", "title": "Amplitude launches AI Copilot", "url": "https://example.com/1", "content": "Amplitude released...", "score": 0.9},
                {"competitor": "Mixpanel", "title": "Mixpanel adds predictive analytics", "url": "https://example.com/2", "content": "Mixpanel announced...", "score": 0.85},
            ]
        },
        confidence=0.4,
        retrieved_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def review_output_success():
    from app.models.worker_output import WorkerOutput
    return WorkerOutput(
        source="reviews",
        status="success",
        findings={
            "pain_points": ["Slow dashboard loading", "Missing export options", "Confusing onboarding"],
            "feature_requests": ["Better API docs", "Slack integration", "Custom dashboards"],
            "overall_sentiment": "neutral",
            "reviews_retrieved": 10,
        },
        confidence=0.72,
        retrieved_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sales_output_success():
    from app.models.worker_output import WorkerOutput
    return WorkerOutput(
        source="sales",
        status="success",
        findings={
            "revenue_trend": "MoM growth of 8% over last 3 months",
            "churn_analysis": "Churn at 3.2%, up from 3-month avg of 2.8%",
            "top_feature": "Analytics dashboard — adopted by 72% of accounts",
            "lost_deal_reason": "Pricing cited in 45% of lost deals",
            "nps_insight": "NPS of 42 — room for improvement in onboarding",
            "key_signal": "Price sensitivity is increasing while churn ticks up",
            "months_analysed": 3,
            "avg_churn_last_3_months": 2.8,
        },
        confidence=1.0,
        retrieved_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def failed_output():
    from app.models.worker_output import WorkerOutput
    return WorkerOutput(
        source="reviews",
        status="failed",
        findings={},
        confidence=0.0,
        error="ChatGoogleGenerativeAIError",
        retrieved_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def all_outputs(web_output_success, review_output_success, sales_output_success):
    return [web_output_success, review_output_success, sales_output_success]


# ---------------------------------------------------------------------------
# Mock LLM response helper
# ---------------------------------------------------------------------------

def make_mock_llm_response(content: str) -> AsyncMock:
    mock_response = MagicMock()
    mock_response.content = content
    mock_ainvoke = AsyncMock(return_value=mock_response)
    return mock_ainvoke
