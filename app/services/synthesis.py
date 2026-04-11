import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.models.recommendation import DraftRecommendation
from app.models.worker_output import WorkerOutput
from app.services.confidence import synthesis_confidence

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are a product strategy analyst. You have received findings from three intelligence sources.
Your job is to synthesise them into a clear, actionable product recommendation.

## Research Query
{query}

## Competitor Focus
{competitors}

## Source Findings

### Web Intelligence (competitor activity)
{web_findings}

### Customer Review Intelligence
{review_findings}

### Internal Sales Intelligence
{sales_findings}

## Instructions
1. Identify **overlapping signals** across sources — these are high-confidence insights. Reinforce them.
2. Identify **contradictory signals** — flag these explicitly. Do NOT silently resolve contradictions.
3. Write an **executive summary** of 2-3 sentences capturing the most important insight.
4. List **key findings per source** with citations where available.
5. Write a specific, **actionable product recommendation** the PM can act on.

{feedback_section}

## Output Format
Respond ONLY with a valid JSON object matching this exact schema:
{{
  "summary": "<2-3 sentence executive summary>",
  "findings_by_source": {{
    "web": "<key findings from web intelligence>",
    "reviews": "<key findings from customer reviews>",
    "sales": "<key findings from sales data>"
  }},
  "overlapping_signals": ["<signal 1>", "<signal 2>"],
  "contradictory_signals": ["<contradiction 1>"],
  "recommendation": "<specific actionable recommendation>"
}}
"""

_FEEDBACK_SECTION = """\
## PM Feedback on Previous Draft
The product manager reviewed the previous recommendation and provided this feedback:
{feedback}

Please address this feedback directly in your revised synthesis.
"""


def _format_findings(output: WorkerOutput) -> str:
    if output.status != "success":
        return f"[Agent {output.source} did not return results — status: {output.status}]"
    return json.dumps(output.findings, indent=2)


def synthesize(
    run_id: int,
    query: str,
    competitors: list[str],
    outputs: list[WorkerOutput],
    feedback: str | None = None,
    rejection_count: int = 0,
) -> DraftRecommendation:
    """
    Call Gemini to synthesise findings from all three worker outputs into a
    DraftRecommendation. Injects PM feedback when re-synthesising after rejection.
    """
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.synthesis_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )

    web_out = next((o for o in outputs if o.source == "web"), None)
    review_out = next((o for o in outputs if o.source == "reviews"), None)
    sales_out = next((o for o in outputs if o.source == "sales"), None)

    feedback_section = (
        _FEEDBACK_SECTION.format(feedback=feedback) if feedback else ""
    )

    prompt = _SYNTHESIS_PROMPT.format(
        query=query,
        competitors=", ".join(competitors),
        web_findings=_format_findings(web_out) if web_out else "[not available]",
        review_findings=_format_findings(review_out) if review_out else "[not available]",
        sales_findings=_format_findings(sales_out) if sales_out else "[not available]",
        feedback_section=feedback_section,
    )

    logger.info("Running synthesis for run_id=%d (rejection_count=%d)", run_id, rejection_count)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Synthesis JSON parse failed: %s", exc)
        raise ValueError(f"Synthesis returned invalid JSON: {exc}") from exc
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", type(exc).__name__)
        raise

    confidence = synthesis_confidence(outputs)

    return DraftRecommendation(
        run_id=run_id,
        summary=parsed.get("summary", ""),
        findings_by_source={
            "web": parsed.get("findings_by_source", {}).get("web", ""),
            "reviews": parsed.get("findings_by_source", {}).get("reviews", ""),
            "sales": parsed.get("findings_by_source", {}).get("sales", ""),
            "overlapping_signals": parsed.get("overlapping_signals", []),
            "contradictory_signals": parsed.get("contradictory_signals", []),
            "recommendation": parsed.get("recommendation", ""),
        },
        confidence=confidence,
        status="pending_approval",
        rejection_count=rejection_count,
    )
