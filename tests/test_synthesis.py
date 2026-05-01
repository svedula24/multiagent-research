"""
Unit tests for confidence scoring and synthesis service.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm_response


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestWebConfidence:
    def test_zero_results(self):
        from app.services.confidence import web_confidence
        assert web_confidence([]) == 0.0

    def test_partial_results(self):
        from app.services.confidence import web_confidence
        assert web_confidence([{}] * 3) == 0.6

    def test_full_results(self):
        from app.services.confidence import web_confidence
        assert web_confidence([{}] * 5) == 1.0

    def test_caps_at_one(self):
        from app.services.confidence import web_confidence
        assert web_confidence([{}] * 10) == 1.0


class TestReviewConfidence:
    def test_empty(self):
        from app.services.confidence import review_confidence
        assert review_confidence([]) == 0.0

    def test_average(self):
        from app.services.confidence import review_confidence
        assert review_confidence([0.8, 0.9, 0.7]) == 0.8

    def test_rounds_to_3dp(self):
        from app.services.confidence import review_confidence
        result = review_confidence([0.8333, 0.7777])
        assert result == round((0.8333 + 0.7777) / 2, 3)


class TestSalesConfidence:
    def test_all_fields_present(self):
        from app.services.confidence import sales_confidence
        row = {"revenue": 100, "churn_rate": 3.0, "new_customers": 10,
               "lost_deals_reason": "price", "top_feature": "analytics", "nps_score": 42}
        assert sales_confidence(row) == 1.0

    def test_all_fields_null(self):
        from app.services.confidence import sales_confidence
        row = {"revenue": None, "churn_rate": None, "new_customers": None,
               "lost_deals_reason": None, "top_feature": None, "nps_score": None}
        assert sales_confidence(row) == 0.0

    def test_partial_fields(self):
        from app.services.confidence import sales_confidence
        row = {"revenue": 100, "churn_rate": None, "new_customers": None,
               "lost_deals_reason": None, "top_feature": None, "nps_score": None}
        assert sales_confidence(row) == round(1 / 6, 3)


class TestSynthesisConfidence:
    def test_all_success(self, all_outputs):
        from app.services.confidence import synthesis_confidence
        score = synthesis_confidence(all_outputs)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # all 3 agents succeeded

    def test_all_failed(self):
        from app.models.worker_output import WorkerOutput
        from app.services.confidence import synthesis_confidence
        from datetime import datetime, timezone
        outputs = [
            WorkerOutput(source=s, status="failed", findings={}, confidence=0.0,
                         retrieved_at=datetime.now(timezone.utc))
            for s in ["web", "reviews", "sales"]
        ]
        assert synthesis_confidence(outputs) == 0.0

    def test_partial_success(self, web_output_success, failed_output, sales_output_success):
        from app.services.confidence import synthesis_confidence
        outputs = [web_output_success, failed_output, sales_output_success]
        score = synthesis_confidence(outputs)
        # source_coverage = 2/3 ≈ 0.667, avg_confidence = (1.0 + 0.4) / 2 = 0.7
        # synthesis = (0.667 + 0.7) / 2 = 0.683
        assert score == 0.683

    def test_formula_correctness(self):
        from app.models.worker_output import WorkerOutput
        from app.services.confidence import synthesis_confidence
        from datetime import datetime, timezone
        outputs = [
            WorkerOutput(source="web", status="success", findings={}, confidence=0.8,
                         retrieved_at=datetime.now(timezone.utc)),
            WorkerOutput(source="reviews", status="success", findings={}, confidence=0.6,
                         retrieved_at=datetime.now(timezone.utc)),
            WorkerOutput(source="sales", status="success", findings={}, confidence=1.0,
                         retrieved_at=datetime.now(timezone.utc)),
        ]
        score = synthesis_confidence(outputs)
        expected = round((1.0 + (0.8 + 0.6 + 1.0) / 3) / 2, 3)
        assert score == expected


# ---------------------------------------------------------------------------
# Synthesis service
# ---------------------------------------------------------------------------

class TestSynthesize:

    async def test_successful_synthesis(self, all_outputs):
        llm_response = json.dumps({
            "summary": "Competitors are investing heavily in AI features.",
            "findings_by_source": {
                "web": "Amplitude launched AI Copilot.",
                "reviews": "Customers want better analytics.",
                "sales": "Revenue growing but churn ticking up.",
            },
            "overlapping_signals": ["AI analytics demand is high"],
            "contradictory_signals": [],
            "recommendation": "Prioritise AI-powered analytics dashboard in next sprint.",
        })

        with patch("app.services.synthesis.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.ainvoke = make_mock_llm_response(llm_response)
            from app.services.synthesis import synthesize
            draft = await synthesize(
                run_id=1,
                query="What are competitors doing in AI analytics?",
                competitors=["Amplitude"],
                outputs=all_outputs,
            )

        assert draft.run_id == 1
        assert draft.status == "pending_approval"
        assert "AI" in draft.summary
        assert 0.0 <= draft.confidence <= 1.0
        assert draft.rejection_count == 0

    async def test_feedback_injected_on_re_synthesis(self, all_outputs):
        llm_response = json.dumps({
            "summary": "Revised: focus on pricing signals.",
            "findings_by_source": {"web": "...", "reviews": "...", "sales": "..."},
            "overlapping_signals": [],
            "contradictory_signals": [],
            "recommendation": "Lower pricing tier to reduce churn.",
        })

        with patch("app.services.synthesis.ChatGoogleGenerativeAI") as MockLLM:
            mock_ainvoke = make_mock_llm_response(llm_response)
            MockLLM.return_value.ainvoke = mock_ainvoke
            from app.services.synthesis import synthesize
            draft = await synthesize(
                run_id=1,
                query="test",
                competitors=["OpenAI"],
                outputs=all_outputs,
                feedback="Focus more on pricing",
                rejection_count=1,
            )

        assert draft.rejection_count == 1
        # Verify feedback was included in the prompt
        call_args = mock_ainvoke.call_args[0][0]
        assert "Focus more on pricing" in call_args

    async def test_strips_markdown_fences(self, all_outputs):
        llm_response = """```json
{
  "summary": "Test summary.",
  "findings_by_source": {"web": "w", "reviews": "r", "sales": "s"},
  "overlapping_signals": [],
  "contradictory_signals": [],
  "recommendation": "Do X."
}
```"""
        with patch("app.services.synthesis.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.ainvoke = make_mock_llm_response(llm_response)
            from app.services.synthesis import synthesize
            draft = await synthesize(
                run_id=1, query="test", competitors=["X"], outputs=all_outputs
            )

        assert draft.summary == "Test summary."

    async def test_raises_on_invalid_json(self, all_outputs):
        with patch("app.services.synthesis.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.ainvoke = make_mock_llm_response("not valid json at all")
            from app.services.synthesis import synthesize
            with pytest.raises(ValueError, match="invalid JSON"):
                await synthesize(
                    run_id=1, query="test", competitors=["X"], outputs=all_outputs
                )


# ---------------------------------------------------------------------------
# Rejection loop counter
# ---------------------------------------------------------------------------

class TestRejectionLogic:

    def test_draft_recommendation_caps_rejection_count(self):
        from app.models.recommendation import DraftRecommendation
        draft = DraftRecommendation(
            run_id=1,
            summary="test",
            confidence=0.7,
            rejection_count=3,
        )
        assert draft.rejection_count == 3

    def test_draft_recommendation_rejects_over_limit(self):
        from app.models.recommendation import DraftRecommendation
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            DraftRecommendation(
                run_id=1, summary="test", confidence=0.7, rejection_count=4
            )
