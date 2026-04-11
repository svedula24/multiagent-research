# AI Research Assistant

A multi-agent AI system that helps product managers make data-driven decisions by autonomously gathering competitor intelligence, analyzing customer feedback, and retrieving internal sales signals — then synthesizing everything into an actionable recommendation with a human approval step.

## Architecture

```
Streamlit UI
    ↓ POST /research/run
FastAPI
    ↓
LangGraph State Machine
    ↓ asyncio.gather (parallel)
┌─────────────────┬──────────────────┬──────────────────┐
│   Web Agent     │  Review Agent    │   Sales Agent    │
│  Tavily Search  │  pgvector + RAG  │  SQL + Gemini    │
└─────────────────┴──────────────────┴──────────────────┘
    ↓ WorkerOutput (Pydantic)
Synthesis Agent (Gemini)
    ↓ DraftRecommendation
Human Approval (Streamlit)
    ↓ approve / reject + feedback
final_recommendations table
```

Three agents run in parallel via `asyncio.gather`. Each returns a typed `WorkerOutput`. The synthesis agent identifies overlapping and contradictory signals, then produces an executive summary and product recommendation. The PM reviews it in Streamlit and can approve or reject (with feedback) — triggering re-synthesis up to 3 times before the run is marked failed.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (state machine + human-in-the-loop) |
| LLMs | Gemini 2.5 Flash (workers + synthesis) |
| Web search | Tavily API via LangChain |
| Vector store | PostgreSQL + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Database driver | psycopg2 (raw SQL, no ORM) |
| Validation | Pydantic v2 |
| API | FastAPI + SlowAPI rate limiting |
| Frontend | Streamlit |
| Observability | LangSmith |
| Evals | RAGAS |
| Testing | pytest + pytest-asyncio |

## Project Structure

```
├── app/
│   ├── main.py                  # FastAPI entrypoint + lifespan
│   ├── config.py                # Settings via pydantic-settings
│   ├── database.py              # psycopg2 connection pool
│   ├── agents/
│   │   ├── manager.py           # asyncio.gather dispatcher
│   │   ├── web_agent.py         # Tavily web search
│   │   ├── review_agent.py      # pgvector semantic search + Gemini
│   │   └── sales_agent.py       # SQL queries + Gemini
│   ├── graph/
│   │   └── research_graph.py    # LangGraph state machine
│   ├── models/
│   │   ├── worker_output.py     # WorkerOutput schema
│   │   ├── recommendation.py    # DraftRecommendation schema
│   │   └── research_run.py      # Request/response schemas
│   ├── routers/
│   │   ├── research.py          # POST /research/run, GET /research/{id}
│   │   └── approval.py          # POST /research/{id}/approve|reject
│   ├── services/
│   │   ├── synthesis.py         # Gemini synthesis prompt + parsing
│   │   ├── confidence.py        # Confidence scoring formulas
│   │   └── embedding.py         # OpenAI embedding service
│   └── tools/
│       ├── tavily_tool.py       # Tavily search wrapper
│       ├── pgvector_tool.py     # Cosine similarity search
│       └── sql_tool.py          # Parameterized SQL wrapper
├── frontend/
│   └── streamlit_app.py         # PM-facing UI
├── scripts/
│   ├── ingest_reviews.py        # Generate + embed 50 synthetic reviews
│   └── ingest_sales.py          # Generate 12 synthetic sales reports
├── sql/
│   └── schema.sql               # 6-table PostgreSQL schema
└── tests/
    ├── conftest.py              # Shared fixtures
    ├── test_agents.py           # Unit tests — agents + manager
    ├── test_synthesis.py        # Unit tests — confidence + synthesis
    └── test_evals.py            # RAGAS faithfulness + answer relevancy
```

## Setup

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + pgvector)
- API keys: Google AI (Gemini), OpenAI, Tavily

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd multiagent-research
python -m venv multiagent-research
source multiagent-research/bin/activate  # Windows: multiagent-research\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```bash
GOOGLE_API_KEY=          # Gemini — gemini.google.com/
OPENAI_API_KEY=          # OpenAI — embeddings only
TAVILY_API_KEY=          # Tavily — web search
LANGSMITH_API_KEY=       # Optional — LangSmith tracing
LANGSMITH_PROJECT=research-assistant
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/research_assistant
LANGCHAIN_TRACING_V2=false   # Set true to enable LangSmith
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. Start the database

```bash
docker compose up -d
```

Apply the schema:

```bash
psql postgresql://postgres:postgres@localhost:5432/research_assistant -f sql/schema.sql
```

### 4. Ingest synthetic data

```bash
python scripts/ingest_reviews.py   # inserts 50 customer reviews with embeddings
python scripts/ingest_sales.py     # inserts 12 monthly sales reports
```

Both scripts are idempotent — they check row counts before inserting.

### 5. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Run the frontend

```bash
streamlit run frontend/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. Enter a research question (e.g. *"What are our competitors doing in AI analytics, and what are customers asking for?"*)
2. Enter competitor names separated by commas (e.g. `Amplitude, Mixpanel, Heap`)
3. Click **Run Research** — the three agents start in parallel
4. Watch agent status badges update as each completes
5. Review the synthesized recommendation with confidence score and per-source findings
6. **Approve** to finalize, or **Reject** with feedback to trigger re-synthesis (up to 3 cycles)

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/research/run` | Submit a research request, returns `run_id` |
| `GET` | `/research/{run_id}` | Poll for status + draft recommendation |
| `POST` | `/research/{run_id}/approve` | Approve the draft |
| `POST` | `/research/{run_id}/reject` | Reject with optional feedback |
| `GET` | `/health` | Database connectivity check |

Rate limits: 20 req/min for write endpoints, 60 req/min for reads.

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Agents

### Web Agent
Searches Tavily for recent competitor activity using the query `"{competitor}" product announcement OR feature launch OR update`. Runs per-competitor searches and aggregates results. Confidence = `min(results / 5, 1.0)`.

### Review Agent
Embeds the research query with OpenAI, performs cosine similarity search against `customer_reviews.embedding`, retrieves the top 10 most relevant reviews, then uses Gemini to extract pain points, feature requests, and overall sentiment. Confidence = average similarity score.

### Sales Agent
Executes parameterized SQL against `sales_reports` to extract revenue trends, churn rate vs 3-month average, top feature adoption, lost deal reasons, and NPS score. Passes results to Gemini for interpretation. Confidence = ratio of non-null fields.

### Synthesis
All three `WorkerOutput` objects are passed to Gemini with a prompt that:
- Identifies overlapping signals across sources (reinforce)
- Flags contradictory signals (explicit, not silently resolved)
- Generates a 2-3 sentence executive summary
- Produces an actionable product recommendation

Synthesis confidence:
```python
source_coverage = successful_agents / 3
avg_confidence  = mean(o.confidence for successful outputs)
final_score     = round((source_coverage + avg_confidence) / 2, 3)
```

## LangGraph State Machine

```
START → dispatch_agents → run_synthesis → await_approval
                                               ↓ approve
                                           finalize → END
                                               ↓ reject
                                          re_synthesize
                                               ↓ (if < 3 cycles)
                                          run_synthesis  (loop)
                                               ↓ (if 3 cycles reached)
                                              END (failed)
```

Nodes that call external services have `RetryPolicy` configured:
- DB writes: 3 attempts, 1s → 2s → 4s backoff
- LLM calls: 3 attempts, 2s → 4s → 8s backoff

The graph uses `MemorySaver` for checkpointing and `interrupt()` for the human-in-the-loop pause at `await_approval`. Each run is keyed by `thread_id = str(run_id)`.

## Running Tests

```bash
# Unit tests (all mocked — no DB or API keys needed)
pytest tests/test_agents.py tests/test_synthesis.py -v

# RAGAS eval suite (requires live DB + API keys)
pytest tests/test_evals.py -v -s
```

The unit test suite covers:
- Web, review, and sales agents with mocked tools
- Retry and timeout behavior
- All confidence scoring formulas
- Synthesis with success / partial failure / all-failure outputs
- Rejection loop counter validation
- Markdown fence stripping from LLM responses

RAGAS evaluates `faithfulness` (does the analysis stay true to the retrieved reviews?) and `answer_relevancy` (is the analysis relevant to the query?) with a threshold of ≥ 0.5.

## Database Schema

Six tables: `customer_reviews` (with `VECTOR(1536)`), `sales_reports`, `research_runs`, `worker_outputs`, `draft_recommendations`, `final_recommendations`. See [`sql/schema.sql`](sql/schema.sql) for full definitions.

The `customer_reviews` embedding column uses an IVFFlat index for approximate nearest-neighbor search:

```sql
CREATE INDEX ON customer_reviews USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
```

## Security

- All secrets loaded from environment variables — never hardcoded
- All SQL queries use parameterized statements via psycopg2
- FastAPI validates and sanitizes inputs via Pydantic before agent dispatch
- LLM outputs validated against Pydantic schemas before downstream use
- Rate limiting on all endpoints via SlowAPI
- `.env` excluded from version control
