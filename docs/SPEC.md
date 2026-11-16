# CareerOps spec

The shared contract is `src/careerops/domain.py`. LLM and embedding interfaces are in `src/careerops/llm.py`.
Each module owns its own tables (prefix `ingest_`, `exp_`, `crm_`, `rank_`) and its own tests.
Modules talk through the domain types and public functions, not through each other's tables.

## Ingestion (`careerops.ingest`, P0)

- Connectors: Greenhouse, Lever, Ashby (public APIs), generic career page, alert-email parser, manual paste.
- Each connector returns `Job` objects. Raw payloads are kept for re-extraction.
- Structured extraction: an LLM fills seniority, remote, salary from the description when the API lacks them. Output is a validated schema.
- Dedupe: blocking on normalized company + title, then similarity on description embeddings,
  then an LLM only for borderline pairs. Sets `canonical_id`.
- Watchlist: `Company.watched` companies are polled and new jobs are reported.
- Evals: a labeled set of duplicate and non-duplicate pairs, and a labeled set for extraction. Report precision and recall.

## Experience and tailoring (`careerops.exp`, P0)

- Experience bank CRUD for `ExperienceItem`. Import from an existing resume with LLM extraction, then user review.
- Retrieval: embeddings plus keyword match pick the items most relevant to a job.
- Tailoring: the LLM rewrites selected items into `ResumeBullet`s. Each bullet must cite `source_item_ids`.
- Grounding check: code verifies every cited id exists. An LLM judge checks each bullet makes no claim beyond its sources.
  Numbers in a bullet must appear in a cited item's text or metrics. Failures set `verified=False` with notes.
- Resume quality checker (P1): requirement coverage report for a job.
- Evals: seeded fabrication tests. The checker must catch invented metrics and skills.

## Application CRM (`careerops.crm`, P0)

- Applications, stage transitions with timestamps, submission records (where, when, how), contacts, deadlines.
- Follow-up manager: default follow-up rules by stage, due and overdue lists.
- Pure code. No LLM needed except optional follow-up message drafts (P1).

## Ranking (`careerops.rank`, P0)

- Hard filters in code: location, seniority range, compensation, avoided companies. Each returns a `FilterResult`.
- Fit score: components computed in code from embeddings and extracted fields. `overall` is a weighted sum in code.
  The LLM writes the explanation and identifies which experience items support each requirement.
- Personalized ranking: starts as rules. Logs `Feedback`. Once enough feedback exists, trains a small
  logistic model over the components and blends it in. Reports when it has too little data.
- Evals: ranking quality (NDCG) on feedback-labeled jobs. Ablations against rules-only.

## Later (P1)

Hidden opportunity signals (funding and hiring news), alerts, cover letters, autofill (prefill only, human submits).

## Integration

A `careerops` CLI (typer) ties the modules together. A small API and web UI come after the CLI works.
