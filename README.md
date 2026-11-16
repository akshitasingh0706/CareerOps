# CareerOps

CareerOps is a tool for running a job search. It finds jobs, scores how well they fit you,
tailors your resume from verified facts about your career, and tracks every application and follow-up.

It is built for one user first. The design keeps room to grow into a product for many users.

## Status

Early development. The shared data model and LLM interfaces exist. Modules are being built.

## Modules

| Module | Package | What it does |
|---|---|---|
| Ingestion | `careerops.ingest` | Pulls jobs from ATS APIs, alert emails, pasted URLs. Normalizes and de-duplicates. |
| Experience and tailoring | `careerops.exp` | Stores verified accomplishments. Builds resumes where every bullet traces to a source. |
| Application CRM | `careerops.crm` | Tracks applications, stages, submissions, contacts, follow-ups. |
| Ranking | `careerops.rank` | Hard filters, fit scoring, and a ranker that learns from your feedback. |

See [docs/SPEC.md](docs/SPEC.md) for the design and [docs/SOURCES.md](docs/SOURCES.md) for where job data comes from and why.

## Design rules

- The LLM reads, extracts, judges and writes. Code does arithmetic, filtering, ranking and dedupe decisions.
- Every LLM output that feeds a decision is a validated pydantic object.
- Every resume claim links to an item in your experience bank. Unsupported claims are rejected.
- Every AI component has an eval, so quality changes are measured.
- Your data stays local. `.env`, `data/` and databases are gitignored.

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env   # then add your ANTHROPIC_API_KEY
pytest
```

## Run it

```bash
docker compose up -d                                  # Postgres, Redis, Mailpit
.venv/bin/uvicorn careerops.api:app --port 8010       # API
(cd web && npm install && npm run dev)                # web app on http://localhost:3000
```

The web app reads the API address from `web/.env.local` (`NEXT_PUBLIC_API_URL=http://localhost:8010`).

CLI:

```bash
careerops watch greenhouse anthropic --name Anthropic
careerops poll
careerops jobs
careerops save <job id>
careerops pipeline
```

The app stores data in SQLite (`data/careerops.db`) for now. Postgres is running for the move to a product database.
Embeddings use an offline hashing model by default. Set `CAREEROPS_EMBEDDER=fastembed` to use `fastembed`,
which downloads a model from Hugging Face on first use.

## License

MIT
