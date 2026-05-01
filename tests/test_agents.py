"""
Unit tests for the three worker agents and the manager dispatcher.
All external calls (Tavily, DB, LLM) are mocked.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm_response


# ---------------------------------------------------------------------------
# Web Agent
# ---------------------------------------------------------------------------

class TestWebAgent:

    async def test_success_returns_worker_output(self):
        mock_results = [
            {"title": "Amplitude AI", "url": "https://a.com", "content": "...", "score": 0.9},
            {"title": "Amplitude Q2", "url": "https://b.com", "content": "...", "score": 0.8},
        ]
        with patch("app.agents.web_agent.search", return_value=mock_results):
            from app.agents.web_agent import run
            result = await run(["Amplitude"])

        assert result.source == "web"
        assert result.status == "success"
        assert result.confidence == 0.4  # 2/5
        assert len(result.findings["results"]) == 2
        assert result.findings["results"][0]["competitor"] == "Amplitude"

    async def test_confidence_maxes_at_1(self):
        mock_results = [{"title": f"r{i}", "url": "", "content": "", "score": 0.9} for i in range(5)]
        with patch("app.agents.web_agent.search", return_value=mock_results):
            from app.agents.web_agent import run
            result = await run(["OpenAI"])

        assert result.confidence == 1.0

    async def test_multiple_competitors_aggregated(self):
        mock_results = [{"title": "t", "url": "", "content": "", "score": 0.5}]
        with patch("app.agents.web_agent.search", return_value=mock_results):
            from app.agents.web_agent import run
            result = await run(["OpenAI", "Google", "Anthropic"])

        # 3 competitors × 1 result each = 3 results total
        assert len(result.findings["results"]) == 3
        sources = {r["competitor"] for r in result.findings["results"]}
        assert sources == {"OpenAI", "Google", "Anthropic"}

    async def test_returns_failed_output_on_error(self):
        with patch("app.agents.web_agent.search", side_effect=Exception("API down")):
            from app.agents import web_agent
            # reload to reset retry state
            result = await web_agent.run(["Amplitude"])

        assert result.source == "web"
        assert result.status == "failed"
        assert result.confidence == 0.0
        assert result.error is not None

    async def test_timeout_returns_timeout_output(self):
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(100)

        with patch("app.agents.web_agent._fetch", side_effect=slow_fetch):
            import app.agents.web_agent as wa
            original_timeout = wa._TIMEOUT
            wa._TIMEOUT = 0.01
            result = await wa.run(["Amplitude"])
            wa._TIMEOUT = original_timeout

        assert result.status == "timeout"


# ---------------------------------------------------------------------------
# Review Agent
# ---------------------------------------------------------------------------

class TestReviewAgent:

    async def test_success_returns_worker_output(self):
        mock_embedding = [0.1] * 1536
        mock_rows = [
            {"id": i, "content": f"Review {i}", "rating": 4, "category": "features",
             "created_at": datetime.now(timezone.utc), "similarity": 0.85}
            for i in range(10)
        ]
        llm_json = json.dumps({
            "pain_points": ["Slow load", "No export", "Bad UI"],
            "feature_requests": ["API docs", "Slack", "Dashboard"],
            "overall_sentiment": "neutral",
        })

        with patch("app.agents.review_agent.embed_text", return_value=mock_embedding), \
             patch("app.agents.review_agent.similarity_search", return_value=mock_rows), \
             patch("app.agents.review_agent.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.ainvoke = make_mock_llm_response(llm_json)
            from app.agents.review_agent import run
            result = await run("What are customers saying about performance?")

        assert result.source == "reviews"
        assert result.status == "success"
        assert result.confidence == 0.85
        assert result.findings["pain_points"] == ["Slow load", "No export", "Bad UI"]
        assert result.findings["reviews_retrieved"] == 10

    async def test_failed_output_on_llm_error(self):
        mock_embedding = [0.1] * 1536
        mock_rows = [
            {"id": 1, "content": "Review", "rating": 3, "category": "features",
             "created_at": datetime.now(timezone.utc), "similarity": 0.7}
        ]
        with patch("app.agents.review_agent.embed_text", return_value=mock_embedding), \
             patch("app.agents.review_agent.similarity_search", return_value=mock_rows), \
             patch("app.agents.review_agent.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
            from app.agents.review_agent import run
            result = await run("test query")

        assert result.status == "failed"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Sales Agent
# ---------------------------------------------------------------------------

class TestSalesAgent:

    def _mock_rows(self):
        return [
            {"period": "2026-04", "revenue": 150000.0, "churn_rate": 3.2,
             "new_customers": 45, "lost_deals_reason": "pricing",
             "top_feature": "analytics", "nps_score": 42, "segment": "mid-market"}
        ]

    async def test_success_returns_worker_output(self):
        mock_rows = self._mock_rows()
        mock_avg = [{"avg_churn": 2.8}]
        llm_json = json.dumps({
            "revenue_trend": "Growing 8% MoM",
            "churn_analysis": "3.2% vs 2.8% avg",
            "top_feature": "analytics",
            "lost_deal_reason": "pricing",
            "nps_insight": "NPS 42",
            "key_signal": "Price sensitivity rising",
        })

        with patch("app.agents.sales_agent.run_query", side_effect=[mock_rows, mock_avg]), \
             patch("app.agents.sales_agent.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.ainvoke = make_mock_llm_response(llm_json)
            from app.agents.sales_agent import run
            result = await run("What do sales trends show?")

        assert result.source == "sales"
        assert result.status == "success"
        assert result.confidence == 1.0  # all 6 fields present
        assert result.findings["top_feature"] == "analytics"

    async def test_no_data_returns_failed(self):
        with patch("app.agents.sales_agent.run_query", side_effect=[[], []]):
            from app.agents.sales_agent import run
            result = await run("test")

        assert result.status == "failed"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class TestManager:

    async def test_dispatch_returns_three_outputs(self, web_output_success,
                                                   review_output_success,
                                                   sales_output_success):
        with patch("app.agents.manager.web_agent.run", AsyncMock(return_value=web_output_success)), \
             patch("app.agents.manager.review_agent.run", AsyncMock(return_value=review_output_success)), \
             patch("app.agents.manager.sales_agent.run", AsyncMock(return_value=sales_output_success)):
            from app.agents.manager import dispatch
            outputs = await dispatch("test query", ["OpenAI"])

        assert len(outputs) == 3
        sources = {o.source for o in outputs}
        assert sources == {"web", "reviews", "sales"}

    async def test_dispatch_proceeds_with_partial_failure(self, web_output_success,
                                                           failed_output,
                                                           sales_output_success):
        with patch("app.agents.manager.web_agent.run", AsyncMock(return_value=web_output_success)), \
             patch("app.agents.manager.review_agent.run", AsyncMock(return_value=failed_output)), \
             patch("app.agents.manager.sales_agent.run", AsyncMock(return_value=sales_output_success)):
            from app.agents.manager import dispatch
            outputs = await dispatch("test query", ["OpenAI"])

        assert len(outputs) == 3
        failed = [o for o in outputs if o.status == "failed"]
        assert len(failed) == 1
