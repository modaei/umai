# UMAI

Ometrics is an oil-field operations platform used to monitor, control, and
report production data. It tracks field readings, tanks, wells, shutdowns,
alarms, work orders, notes, emails, and production reports across a selected
site.

UMAI is the read-only AI agent for Ometrics. It interprets oil-field operations 
questions, chooses the appropriate tool or data path, retrieves the required 
operational context, and produces grounded answers from the retrieved data. 
The agent is built with LangChain, LangGraph and an OpenAI-compatible chat
model, and it can call Ometrics, Omreports, RAG, and validated SQL tools
depending on the user's request.

The Ometrics application uses UMAI through a local FastAPI `/chat`
endpoint. UMAI is not intended to run independently. It is an AI agent layer
for an existing Ometrics deployment, with Ometrics and Omreports remaining the
systems of record.

## Responsibilities

UMAI is responsible for:

- Receiving chat requests from Ometrics and maintaining server-side
  conversation history by `conversation_id`, `user_id`, and `site_id`.
- Using an OpenAI-compatible chat API through LangChain. The model, API key, and
  base URL are configurable, so it can run with OpenAI directly or a compatible
  provider such as OpenRouter.
- Acting as a tool-using operational agent: interpreting user intent, selecting
  the best available tool, passing scoped arguments, and synthesizing the result
  into a user-facing answer.
- Running Omreports functions for report questions, including production,
  sales, gas, water, injection, allocation, and monthly battery summaries.
- Reading Ometrics MySQL data for raw readings, missing readings, comparisons,
  tank volumes, well shutdowns, shutdown-cause summaries, current long
  shutdowns, well timelines, alarms, notes, and work-history context.
- Running tightly validated read-only SQL against allowlisted operational
  tables when the specific report, reading, shutdown, timeline, work-order, or
  capability tools are not the best fit for the question.
- Answering questions about Ometrics capabilities using curated capability
  documents in `knowledge/capabilities`.
- Searching operational text with RAG. Source records are read from MySQL during
  indexing, embedded, and stored in a Postgres + pgvector index.
- Producing answers grounded in retrieved operational data, tool results, and
  curated Ometrics capability guidance.
- Enforcing read-only behavior. UMAI can retrieve and summarize data, and it can
  prepare navigation to existing Ometrics forms where supported, but it does not
  create records, update records, send emails, acknowledge alarms, control
  equipment, or export files.

## Runtime Dependencies

UMAI is not a standalone application. It is designed to run as part of the
Ometrics software bundle.

At runtime it depends on:

- Ometrics MySQL database for site data, readings, shutdowns, notes, alarms,
  well history, and conversation storage.
- Omreports service for calculated production, sales, gas, water, injection,
  allocation, and battery reports.
- A configured LLM provider through an OpenAI-compatible API.
- Postgres + pgvector for operational-text RAG, if RAG questions should be
  supported.

Without the Ometrics database and Omreports service, UMAI can start, but most
domain tools will be unavailable or return errors.

## Architecture

- `FastAPI` serves `/health` and `/chat` for Ometrics. The API accepts only
  localhost clients.
- `LangChain` runs the agent loop: understand the request, decide whether a
  tool is needed, call the selected tool with scoped arguments, then compose the
  final answer.
- Deterministic routing handles common high-confidence workflows before a full
  LLM tool-selection loop is needed.
- `omreports` remains the source for calculated report results.
- Ometrics MySQL remains the source of operational records.
- Postgres + pgvector stores the derived operational-text vector index.
- Conversation messages are stored in the Ometrics database through UMAI's
  conversation repository.

Typical flow:

```text
Ometrics chat request
        │
        ▼
UMAI AI agent
        │
        ├─ deterministic router, when a safe direct path exists
        └─ LangChain LLM tool-selection loop
                 │
                 ▼
        Ometrics / Omreports / RAG / validated SQL tools
                 │
                 ▼
        grounded final answer returned to Ometrics
```

The public `/chat` response contains only:

```json
{
  "conversation_id": "uuid",
  "answer": "assistant response"
}
```

Tool calls, timing statistics, and RAG source snippets are internal and are not
returned by the API. Assistant message metadata, including compact tool, RAG,
and SQL audit records, is stored in `ai_messages.info`.

## Setup

Run the following UMAI commands from the UMAI repository root. Create a virtual
environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
cp .env.example .env
```

Set the LLM provider values in `.env`.

Set `LOG_LEVEL` to control UMAI logging across the API, CLI commands, and RAG
event processor. Supported values are:

```bash
LOG_LEVEL=CRITICAL
LOG_LEVEL=ERROR
LOG_LEVEL=WARNING
LOG_LEVEL=INFO
LOG_LEVEL=DEBUG
LOG_LEVEL=NOTSET
```

For OpenAI directly:

```bash
LLM_API_KEY=your-openai-key
LLM_MODEL=gpt-5-mini
LLM_BASE_URL=https://api.openai.com/v1
```

For OpenRouter:

```bash
LLM_API_KEY=your-openrouter-key
LLM_MODEL=openai/gpt-5-mini
LLM_BASE_URL=https://openrouter.ai/api/v1
```

Configure the Ometrics MySQL connection for reading operational data and storing
conversation history:

```bash
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=ometrics_read_user
DB_PASSWORD=secret
DB_NAME=ometrics
```

For direct operational SQL, configure a separate database user that has only
`SELECT` privileges on the allowlisted operational tables. UMAI also validates
SQL shape, table names, columns, site scoping, and row limits before execution,
but database permissions are the final safety boundary:

```bash
OPERATIONAL_SQL_DB_USER=ometrics_ai_readonly
OPERATIONAL_SQL_DB_PASSWORD=secret
OPERATIONAL_SQL_MAX_ROWS=100
OPERATIONAL_SQL_TIMEOUT_SECONDS=10
```

Configure Omreports:

```bash
OMREPORTS_API_URL=http://127.0.0.1:50008/report/
MONITORING_DATA_API_URL=http://metrics1.ultimatesys.com/render
ROD_PUMP_TIMEOUT_SECONDS=20
ROD_PUMP_MAX_DATA_POINTS=96
ROD_PUMP_BATCH_WORKERS=2
# Leave unset until a paraffin model passes the validation gate.
ROD_PUMP_MODEL_METADATA=
OMREPORTS_TIMEOUT_SECONDS=30
MAX_REPORT_DAYS=366
```

## Operational RAG

Operational-text RAG uses a derived vector index backed by Postgres + pgvector.
UMAI reads source records from MySQL and writes only the normalized chunks and
embeddings to the vector database.

Configure the vector database:

```bash
VECTOR_DB_HOST=127.0.0.1
VECTOR_DB_PORT=5432
VECTOR_DB_USER=ometrics
VECTOR_DB_PASSWORD=
VECTOR_DB_NAME=ometrics
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSIONS=1536
```

`CREATE EXTENSION vector` must be run once by a privileged Postgres user inside
the same database configured by `VECTOR_DB_NAME`:

```sql
CREATE EXTENSION vector;
```

Then create/update the vector schema and index operational text:

```bash
umai-vector-db-migrate
umai-index-operational-text --site-id 4 --from-date 2026-01-01 --reset-site
```

The indexer includes general notes, chart notes, work orders and notes,
shutdown comments, downtime codes, well-test comments, reading comments, alarm
logs, and well history records.

Ometrics keeps this index current by writing model-change events into the shared
`ai_rag_index_events` MySQL outbox table. Run `umai-process-rag-index-events`
from cron or a systemd timer every few minutes; each run takes a MySQL lock,
processes currently due events, deletes successful rows, and exits. A nightly
cron job can also call the `umai-index-operational-text` command directly for
rolling repair refreshes.

## Running Services

Start Omreports. The Ometrics software bundle keeps the Omreports repository
next to the UMAI repository:

```bash
source venv/bin/activate
python3 -m uvicorn api:app --host 127.0.0.1 --port 50008
```

Start UMAI API for Ometrics:

```bash
source .venv/bin/activate
python3 -m uvicorn umai.api.app:app --host 127.0.0.1 --port 50009
```

Process queued UMAI RAG index events periodically:

```bash
source .venv/bin/activate
umai-process-rag-index-events
```

## Example Questions

- How much gas was flared during May?
- Compare Battery 6 oil production for each month in 2026.
- Which readings are missing on 2026-05-20?
- Compare mixed tank readings on 2026-05-09 and 2026-05-10.
- What happened with 5144H on 2026-06-14?
- What can you tell me about 4293?
- What has been the main cause of shutdowns in the last 30 days?
- Build a timeline for Well 4048 this year.
- How do I register a LACT reading?
- How can I know how much each well contributed to oil production?

## Tests

```bash
source .venv/bin/activate
pytest -q
```

The unit tests use fakes and in-memory databases where possible. They do not
require a live LLM API, Omreports service, MySQL database, or vector database.

## Local Evaluation

UMAI includes a developer-only live evaluation command:

```bash
pip install ".[dev]"
umai-evaluate
```

This command uses the real local UMAI configuration: the configured UMAI LLM for
answers, Ometrics MySQL database, Omreports service, operational SQL user, and
Postgres + pgvector RAG index. For DeepEval judge calls, UMAI maps `LLM_API_KEY`
and `LLM_BASE_URL` into OpenAI-compatible environment variables when they are not
already set. It is not part of normal CI and may consume LLM/API credits.

Useful filters:

```bash
umai-evaluate --limit 3
umai-evaluate --category rag
umai-evaluate --case-id report-gas-flared-may-2026
umai-evaluate --failed-only
```

Use `--current-date YYYY-MM-DD` to supply a default date for cases that do not
define `current_date`. If a case defines `current_date`, the case value wins.

Before running cases, the command validates local DB, vector DB, and Omreports
connectivity. Results are written to `eval-results/umai-eval-*.json`. Each result
contains only:

- `case.id` and `case.question`
- `passed`
- `answer`
- `tool_calls`
- `failed_checks`
- `failed_metrics`

Passing deterministic checks and passing DeepEval metrics are omitted from the
report to keep files small. Timing stats are intentionally not written to eval
reports.

Use `--failed-only` when running larger suites if the report file should include
only failed cases. The top-level `passed` and `total` counts still refer to all
evaluated cases; only the `results` array is filtered.

Evaluation cases live in `src/umai/evals/cases/core.json`. Important fields:

- `expected_tool_calls`: verifies both the tool name and selected argument
  values. This replaces the older `required_tools` style check.
- `forbidden_tools`, `required_phrases`, and `forbidden_phrases`: deterministic
  answer/tool assertions.
- `deterministic: true`: skips DeepEval judge calls for cases where tool calls,
  arguments, and phrases fully define success. This reduces runtime and token
  cost for data-entry and routing regressions.
- `metrics`: DeepEval metrics to run for non-deterministic cases. Supported
  values include `answer_relevancy`, `correctness`, `faithfulness`, and
  `contextual_relevancy`.
- `expected_output`: reference behavior used by LLM-judged correctness-style
  metrics. It is ignored for `deterministic: true` cases.

`expected_tool_calls[].arguments` supports dotted paths and simple matchers:

```json
{
  "tool": "analyze_well_tests",
  "arguments": {
    "analysis_mode": "recent_tests",
    "group_by": "well",
    "well_name": {"contains": "4048"},
    "test_count": 3
  }
}
```

Available matchers are `contains`, `contains_all`, `one_of`, `present`, and
`absent`.
