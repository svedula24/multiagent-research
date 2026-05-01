"""
RAGAS evaluation suite for the review agent RAG pipeline.

Evaluates:
  - faithfulness:      Does the generated analysis stay true to the retrieved reviews?
  - answer_relevancy:  Is the analysis relevant to the research query?

Requires:
  - Running postgres with populated customer_reviews table
  - Valid GOOGLE_API_KEY and OPENAI_API_KEY in .env

Run with:
    pytest tests/test_evals.py -v -s
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

RAGAS_SCORE_THRESHOLD = 0.5

EVAL_QUERIES = [
    "What are customers saying about onboarding experience and getting started?",
    "What performance issues are customers reporting?",
    "What new features are customers requesting?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_db():
    from app.config import get_settings
    from app.database import init_pool
    settings = get_settings()
    try:
        init_pool(settings.database_url)
    except Exception:
        pass  # already initialised


def retrieve_reviews(query: str, top_k: int = 5) -> list[str]:
    from app.services.embedding import embed_text
    from app.tools.pgvector_tool import similarity_search
    embedding = embed_text(query)
    rows = similarity_search(embedding, top_k=top_k)
    return [r["content"] for r in rows]


async def generate_analysis(query: str, contexts: list[str]) -> str:
    from app.config import get_settings
    from langchain_google_genai import ChatGoogleGenerativeAI
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.worker_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    review_text = "\n\n".join(f"[Review {i+1}]: {c}" for i, c in enumerate(contexts))
    prompt = (
        f"Research question: {query}\n\n"
        f"Customer reviews:\n{review_text}\n\n"
        "Provide a concise 2-3 sentence analysis of the key themes relevant to the question."
    )
    response = await llm.ainvoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# RAGAS eval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_review_rag_pipeline_ragas():
    """
    Evaluate the review RAG pipeline using RAGAS faithfulness and answer relevancy.
    Uses await aevaluate() to avoid nested event loop conflicts.
    """
    from datasets import Dataset
    from ragas import aevaluate
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import OpenAIEmbeddings
    from app.config import get_settings

    setup_db()
    settings = get_settings()

    questions, answers, contexts_list = [], [], []

    for query in EVAL_QUERIES:
        contexts = retrieve_reviews(query, top_k=5)
        if not contexts:
            pytest.skip("No reviews in DB — run scripts/ingest_reviews.py first")

        analysis = await generate_analysis(query, contexts)
        questions.append(query)
        answers.append(analysis)
        contexts_list.append(contexts)

    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts_list,
    })

    ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=settings.worker_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    ))

    result = await aevaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=True,
        raise_exceptions=False,
    )

    scores_df = result.to_pandas()
    print("\n=== RAGAS Evaluation Results ===")
    print(scores_df[["faithfulness", "answer_relevancy"]].to_string())

    mean_scores = scores_df[["faithfulness", "answer_relevancy"]].mean()
    print("\nMean scores:")
    for metric_name, score in mean_scores.items():
        status = "✅" if score >= RAGAS_SCORE_THRESHOLD else "❌"
        print(f"  {status} {metric_name}: {score:.3f} (threshold: {RAGAS_SCORE_THRESHOLD})")

    for metric_name, score in mean_scores.items():
        assert score >= RAGAS_SCORE_THRESHOLD, (
            f"RAGAS {metric_name} score {score:.3f} below threshold {RAGAS_SCORE_THRESHOLD}"
        )
