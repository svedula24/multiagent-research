import logging

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_text(text: str) -> list[float]:
    """
    Embed a text string using OpenAI text-embedding-3-small (1536 dims).

    Args:
        text: The text to embed. Will be truncated to 8191 tokens by the API if needed.

    Returns:
        List of 1536 floats representing the embedding vector.
    """
    settings = get_settings()
    client = _get_client()

    try:
        response = client.embeddings.create(
            input=text,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        vector = response.data[0].embedding
        logger.info("Embedded text of length %d → vector dim %d", len(text), len(vector))
        return vector
    except Exception as exc:
        logger.error("Embedding failed: %s", type(exc).__name__)
        raise
