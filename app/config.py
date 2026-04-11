import logging
from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    google_api_key: str = Field(..., description="Google Gemini API key")

    # Embeddings
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")

    # Web search
    tavily_api_key: str = Field(..., description="Tavily search API key")

    # Observability
    langsmith_api_key: str = Field("", description="LangSmith API key")
    langsmith_project: str = Field("research-assistant", description="LangSmith project name")
    langchain_tracing_v2: bool = Field(False, description="Enable LangChain tracing")
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", description="LangChain endpoint"
    )

    # Database
    database_url: str = Field(
        "postgresql://postgres:postgres@localhost:5432/research_assistant",
        description="PostgreSQL connection URL",
    )

    # Agent model configuration
    worker_model: str = Field("gemini-2.0-flash", description="Gemini model for worker agents")
    synthesis_model: str = Field("gemini-1.5-pro", description="Gemini model for synthesis/manager")
    embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    embedding_dimensions: int = Field(1536, description="Embedding vector dimensions")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        logger.error("Configuration error: %s", e)
        raise
