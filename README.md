# AI Research Assistant

A multi-agent AI system that helps product managers make data-driven decisions by autonomously gathering competitor intelligence, analyzing customer feedback, and retrieving internal sales signals — then synthesizing everything into an actionable recommendation with a human approval step.

## Architecture

```
Streamlit UI
    ↓ POST /research/run
FastAPI (SlowAPI rate limiting)
    ↓
LangGraph State Machine
    ↓ asyncio.gather (parallel, 30s timeout per agent)
┌─────────────────┬──────────────────┬──────────────────┐
│   Web Agent     │  Review Agent    │   Sales Agent    │
│  Tavily Search  │  pgvector + RAG  │  SQL + Gemini    │
└─────────────────┴──────────────────┴──────────────────┘
    ↓ WorkerOutput (Pydantic)
Synthesis Agent (Gemini)
    ↓ DraftRecommendation
Human Approval (Streamlit)
    ↓ approve / reject + feedback (up to 3 cycles)
final_recommendations table
```

Three agents run in parallel via `asyncio.gather`. Each returns a typed `WorkerOutput`. The synthesis agent identifies overlapping and contradictory signals, then produces an executive summary and product recommendation. The PM reviews it in Streamlit and can approve or reject with feedback — triggering re-synthesis using the original agent outputs (agents do not re-run). Maximum 3 rejection cycles before the run is marked failed.

## Agent Architecture

### Web Agent

```
competitors list
    ↓
Per-competitor: "{competitor} product announcement OR feature launch OR update"
    ↓ Tavily API (max_results=5 per competitor)
Aggregate results → [{title, url, content, score, competitor}]
    ↓
WorkerOutput(source="web")
confidence = min(total_results / 5, 1.0)
```

Retry: 3 attempts with exponential backoff (0s → 2s → 4s). Timeout: 30s per run.
On failure returns `status="failed"`, `confidence=0.0`.

---

### Review Agent

```
research query
    ↓ OpenAI text-embedding-3-small (1536 dims)
query embedding
    ↓ pgvector cosine similarity search (top 10 reviews)
retrieved reviews + similarity scores
    ↓ Gemini (worker model)
{pain_points: [...], feature_requests: [...], overall_sentiment: ...}
    ↓
WorkerOutput(source="reviews")
confidence = avg(cosine_similarity scores)
```

Retry: 3 attempts with exponential backoff (0s → 2s → 4s). Timeout: 30s per run.
The IVFFlat index on `customer_reviews.embedding` enables approximate nearest-neighbor search.

---

### Sales Agent

```
research query context
    ↓ Parameterized SQL → sales_reports table
[last 3 months rows, avg churn query]
    ↓ Gemini (worker model)
{revenue_trend, churn_analysis, top_feature, lost_deal_reason, nps_insight, key_signal}
    ↓
WorkerOutput(source="sales")
confidence = non_null_fields / 6
```

Retry: 3 attempts with exponential backoff (0s → 2s → 4s). Timeout: 30s per run.
All SQL uses parameterized psycopg2 queries — no string concatenation.

---

### Manager + Synthesis

```
[WebOutput, ReviewOutput, SalesOutput]  ← asyncio.gather result
    ↓
Gemini synthesis prompt:
  - overlapping signals (reinforce)
  - contradictory signals (flag explicitly, never silently resolve)
  - 2-3 sentence executive summary
  - findings per source with citations
  - actionable product recommendation
    ↓
DraftRecommendation → draft_recommendations table
```

Synthesis confidence:
```python
source_coverage = successful_agents / 3
avg_confidence  = mean(o.confidence for successful outputs)
final_score     = round((source_coverage + avg_confidence) / 2, 3)
```

A failed agent contributes `0.0` to confidence but does not block synthesis — the prompt notes the missing source explicitly.

---

### LangGraph State Machine

```
START → dispatch_agents → run_synthesis → await_approval
                                               ↓ approve
                                           finalize → END
                                               ↓ reject + feedback
                                          re_synthesize
                                               ↓ rejection_count < 3
                                          run_synthesis  (loop back)
                                               ↓ rejection_count == 3
                                              END (failed)
```

- `await_approval` uses LangGraph `interrupt()` to pause the graph — the PM's approve/reject action resumes it via `Command(resume=...)`
- `MemorySaver` checkpoints graph state in memory keyed by `thread_id = str(run_id)`
- DB write nodes use `RetryPolicy(max_attempts=3, backoff=2x)`, LLM nodes use `RetryPolicy(max_attempts=3, initial_interval=2s)`

---

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
│   ├── main.py                  # FastAPI entrypoint + LangSmith setup
│   ├── config.py                # Settings via pydantic-settings
│   ├── database.py              # psycopg2 connection pool
│   ├── agents/
│   │   ├── manager.py           # asyncio.gather dispatcher
│   │   ├── web_agent.py         # Tavily web search + retry/timeout
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
│   │   ├── synthesis.py         # Gemini synthesis prompt + JSON parsing
│   │   ├── confidence.py        # Confidence scoring formulas
│   │   └── embedding.py         # OpenAI embedding service
│   └── tools/
│       ├── tavily_tool.py       # Tavily search wrapper
│       ├── pgvector_tool.py     # Cosine similarity search
│       └── sql_tool.py          # Parameterized SQL wrapper
├── frontend/
│   └── streamlit_app.py         # PM-facing UI with approval flow
├── scripts/
│   ├── ingest_reviews.py        # Generate + embed 50 synthetic reviews
│   └── ingest_sales.py          # Generate 12 monthly sales reports
├── sql/
│   └── schema.sql               # PostgreSQL schema (6 tables)
└── tests/
    ├── conftest.py              # Shared fixtures (all external deps mocked)
    ├── test_agents.py           # Unit tests — agents + manager
    ├── test_synthesis.py        # Unit tests — confidence + synthesis
    └── test_evals.py            # RAGAS faithfulness + answer relevancy
```

## Setup

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + pgvector)
- API keys: Google AI (Gemini), OpenAI, Tavily
- Optional: LangSmith account for tracing

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd multiagent-research
python -m venv multiagent-research
source multiagent-research/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```bash
GOOGLE_API_KEY=          # Gemini — aistudio.google.com
OPENAI_API_KEY=          # OpenAI — embeddings only
TAVILY_API_KEY=          # Tavily — web search
LANGSMITH_API_KEY=       # LangSmith — leave empty to disable tracing
LANGSMITH_PROJECT=research-assistant
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/research_assistant
```

LangSmith tracing is automatically enabled when `LANGSMITH_API_KEY` is set. No other changes needed — the app configures the required LangChain environment variables at startup.

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

If LangSmith is configured, you will see:
```
LangSmith tracing enabled — project=research-assistant
```

### 6. Run the frontend

```bash
streamlit run frontend/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Usage

1. Enter a research question (e.g. *"What are our competitors doing in AI analytics, and what are customers asking for?"*)
2. Enter competitor names separated by commas (e.g. `Amplitude, Mixpanel, Heap`)
3. Click **Run Research** — the three agents start in parallel
4. Watch agent status badges update as each completes (green = success, red = failed, orange = timeout)
5. Review the synthesized recommendation with confidence score and per-source findings tabs
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

## LangSmith Observability

With a valid `LANGSMITH_API_KEY`, every research run produces a trace at `smith.langchain.com`. Each trace shows:

- **LangGraph node tree** — `dispatch_agents` → `run_synthesis` → `await_approval` → `finalize` with per-node latency
- **Parallel agent calls** — the three Gemini/Tavily calls inside `dispatch_agents` shown side by side
- **Full prompts and responses** — the exact synthesis prompt with injected findings, and Gemini's raw JSON response
- **Token counts and cost** — per LLM call and total for the run

To enable: create a project named `research-assistant` at `smith.langchain.com`, generate an API key, and set `LANGSMITH_API_KEY` in `.env`.

## Running Tests

```bash
# Unit tests (all mocked — no DB or API keys needed)
pytest tests/test_agents.py tests/test_synthesis.py -v

# RAGAS eval suite (requires live DB + API keys)
pytest tests/test_evals.py -v -s
```

**Unit tests (31 tests)** cover:
- Web, review, and sales agents with mocked Tavily / pgvector / LLM
- Retry and timeout behavior per agent
- All confidence scoring formulas (web, review, sales, synthesis)
- Synthesis with success / partial failure / all-failure outputs
- Rejection loop counter validation (max 3, Pydantic-enforced)
- Markdown fence stripping from Gemini responses

**RAGAS eval** runs the full review RAG pipeline against the live database and scores:

| Metric | What it measures | Threshold | Typical score |
|---|---|---|---|
| Faithfulness | Does the analysis stay grounded in the retrieved reviews? | ≥ 0.5 | ~0.72 |
| Answer Relevancy | Is the analysis relevant to the original query? | ≥ 0.5 | ~0.82 |

RAGAS uses Gemini as the judge LLM and OpenAI embeddings for relevancy scoring. It internally decomposes the generated analysis into atomic statements and verifies each against the retrieved context.

## Database Schema

Six tables: `customer_reviews` (with `VECTOR(1536)`), `sales_reports`, `research_runs`, `worker_outputs`, `draft_recommendations`, `final_recommendations`. See [`sql/schema.sql`](sql/schema.sql) for full definitions.

The `customer_reviews` embedding column uses an IVFFlat index for approximate nearest-neighbor search:

```sql
CREATE INDEX ON customer_reviews USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

`draft_recommendations` uses `ON CONFLICT (run_id) DO UPDATE` — each re-synthesis overwrites the previous draft in place, keeping one row per run.

## Security

- All secrets loaded from environment variables — never hardcoded
- All SQL queries use parameterized psycopg2 statements — no string concatenation
- FastAPI validates and sanitizes inputs via Pydantic before agent dispatch
- LLM outputs validated against Pydantic schemas before downstream use
- Rate limiting on all endpoints via SlowAPI
- `.env` excluded from version control
