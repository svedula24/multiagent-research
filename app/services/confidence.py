from app.models.worker_output import WorkerOutput


def web_confidence(results: list[dict]) -> float:
    """
    Confidence based on number of Tavily results returned.
    5 or more results → 1.0; scales linearly below that.
    """
    return round(min(len(results) / 5, 1.0), 3)


def review_confidence(similarity_scores: list[float]) -> float:
    """
    Confidence based on average cosine similarity of retrieved reviews.
    Scores are already in [0, 1] from pgvector (1 - cosine_distance).
    """
    if not similarity_scores:
        return 0.0
    return round(sum(similarity_scores) / len(similarity_scores), 3)


def sales_confidence(row: dict) -> float:
    """
    Confidence based on completeness of the sales data fields.
    Each non-null field contributes equally.
    """
    fields = [
        row.get("revenue"),
        row.get("churn_rate"),
        row.get("new_customers"),
        row.get("lost_deals_reason"),
        row.get("top_feature"),
        row.get("nps_score"),
    ]
    filled = sum(1 for f in fields if f is not None)
    return round(filled / len(fields), 3)


def synthesis_confidence(outputs: list[WorkerOutput]) -> float:
    """
    Combined confidence across all three worker outputs.

    Formula (from spec):
        source_coverage = successful_count / 3
        avg_confidence  = mean confidence of successful outputs
        synthesis       = (source_coverage + avg_confidence) / 2
    """
    successful = [o for o in outputs if o.status == "success"]
    source_coverage = len(successful) / 3
    avg_confidence = (
        sum(o.confidence for o in successful) / len(successful) if successful else 0.0
    )
    return round((source_coverage + avg_confidence) / 2, 3)
