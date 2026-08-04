# Health & Product Assistant

Implementation of the target architecture in [arch.md](arch.md), built for cPanel
shared hosting: **MySQL only — no Redis, no Docker, no broker.**

## What runs where

| arch.md calls for | Here | Why |
|---|---|---|
| Postgres + pgvector | MySQL + `VARBINARY` embeddings scored with numpy | The host has MySQL (3306 open, 5432 closed). Brute-force cosine is sub-millisecond at this corpus size. |
| Postgres FTS (`tsvector`) | InnoDB `FULLTEXT` + `MATCH…AGAINST` | Lexical recall for tokens embeddings blur — `TDEE`, `BMR`, `INCI`. |
| Redis L2 cache | `cache_entry` table | Shared across replicas, TTL swept by the worker. |
| Redis job queue / Celery | `job` table, claimed by token | Atomic on any MySQL/MariaDB; no `SKIP LOCKED` needed. |
| Redis rate limits | `rate_limit_bucket` table | Fixed-window counters. |
| Redis locks | MySQL `GET_LOCK()` | Native advisory locks. |
| S3 / GCS | Filesystem under `BLOB_DIR` + `blob` rows | Keep it outside `public_html`. |

Swapping any of these back is a change behind the existing interface, not a rewrite.

## One endpoint

Everything is `POST /`, discriminated by `action` (default `chat`). `GET /` returns
status. This matches the shape the old `app.py` already exposes, so clients keep
the same URL.

```jsonc
POST /
Authorization: Bearer <jwt>

{ "action": "chat", "message": "how much protein should I eat?", "session_id": "..." }
```

| Action | Does |
|---|---|
| `chat` *(default)* | Runs a turn through the graph and returns an `AnswerPayload` |
| `upload` | Stores attachments, returns handles. Bytes enter here and nowhere else |
| `scan` | Upload + analyse in one call — the scanner-screen path |
| `sync` | Writes profile / metrics / nutrition / activities / medical — **replaces `csv_health_data` in the request body** |
| `context` | Shows exactly what the assistant can see about you |
| `consent` | `list` / `grant` / `revoke` per data scope |
| `memory` | `list` / `remember` / `forget` — user-visible and editable |
| `history` | Recent turns for a session |
| `job_status`, `job_enqueue` | Async job polling and submission |
| `admin.cache`, `admin.faq`, `admin.jobs` | Operator surface (admin scope required) |

The response envelope is always `{"success": bool, "request_id": str, ...}`.

## The scan pipeline

```
capture → decode barcode → identify product → OCR label →
parse panel → resolve to chemical ids → hazard rules → personal risk → verdict
```

**The runtime path touches zero external toxicology APIs.** the old `app.py` fans
out to 6 upstream APIs plus an LLM call *per ingredient*, inside the request —
roughly 90 network calls for one scan. This does at most two (a product lookup
and an OCR call, both cached) and a handful of batched local queries, regardless
of panel length.

What holds the safety guarantees:

- **Rules assign hazard levels, never the model.** `hazard_rule` rows are
  declarative and versioned; every finding records which rules fired, so a
  verdict is reproducible and auditable.
- **The verdict is a four-value enum.** Rules choose it; `guard_out` re-checks
  that what ships matches what the rules produced, and that no ingredient named
  in the output is absent from the structured findings. That check is in place
  *before* the LLM explanation layer lands, not retrofitted after.
- **Unresolved tokens are surfaced, never dropped or silently researched.** They
  become a `chemical_research` job (idempotent per token) and the response says
  how many are outstanding.
- **Coverage gates optimism only.** A clean result over a panel we mostly could
  not read returns `Insufficient data`, not `Generally suitable`.
- **All outbound HTTP goes through the fetch broker** — allowlist, DNS checked
  before connect, private/link-local/metadata ranges refused, redirects
  re-validated per hop. This closes the SSRF in the old `app.py` (line 15).

Seed the knowledge base:

```bash
python scripts/seed_chemicals.py    # starter dossiers, hazard rules, cross-reactants
```

`opencv-python-headless` and `zxing-cpp` are imported lazily. Without them the
API still boots and barcode scanning degrades to "photograph the panel".

## Deploying to cPanel

**1. Upload** the repo outside `public_html` (e.g. `~/fitness-api`).

**2. Create the Python app** — cPanel → *Setup Python App*:
- Python 3.10+, application root `~/fitness-api`
- Startup file `passenger_wsgi.py`, entry point `application`
- Then "Run Pip Install" against `requirements.txt`

`passenger_wsgi.py` bridges ASGI→WSGI with `a2wsgi`. Passenger cannot serve
FastAPI directly — it speaks WSGI, and FastAPI is ASGI.

**3. Configure** — copy `.env.example` to `.env` and fill in at minimum:

```
DB_PASSWORD=...
JWT_SECRET=...          # python -c "import secrets; print(secrets.token_urlsafe(48))"
OPENAI_API_KEY=...
CORS_ORIGINS=https://yourapp.com
```

**4. Grant database access.** In cPanel → *MySQL Databases*, confirm
`movenetics_fitness` (user) is granted ALL on `movenetics_fitness` (database). To
run migrations from your laptop rather than the server, add your IP under
*Remote MySQL*.

**5. Create the schema:**

```bash
python scripts/init_db.py --check     # connectivity and config first
python scripts/init_db.py             # applies migrations
python scripts/seed_faq.py            # starter FAQ set
```

If you imported `packages/storage/migrations/*.sql` by hand (phpMyAdmin), the
tables exist but `schema_migration` is empty. Record them and confirm nothing
was dropped mid-import:

```bash
python scripts/init_db.py --mark-applied   # records them, then verifies
python scripts/init_db.py --verify         # 31 tables expected
```

> **If you imported before 2026-08-04, re-import.** Three columns used MySQL
> reserved words (`ocr_result.text`, `evidence_chunk.text`,
> `evidence_document.year`), so those `CREATE TABLE`s failed with ERROR 1064 and
> the tables do not exist. The `blob` table was also renamed to `blob_object`
> for the same reason. Re-running the fixed files is safe — every statement is
> `CREATE TABLE IF NOT EXISTS` — then drop the stale table if it was created:
> `DROP TABLE IF EXISTS \`blob\`;`
>
> `python scripts/check_ddl.py` guards against this class of bug returning. It
> is worth running in CI: reserved-word collisions are invisible to every
> Python test, because the SQL is just a string until it reaches the server.

**6. Schedule the worker** — cPanel → *Cron Jobs*, every minute:

```
cd ~/fitness-api && /home/USER/virtualenv/fitness-api/3.11/bin/python -m apps.worker.worker --once
```

Overlapping ticks are safe: the worker takes a MySQL advisory lock and a second
process exits rather than double-processing. If your plan allows a persistent
process, drop `--once` and run it as a daemon instead.

**7. Verify:**

```bash
curl https://your-domain/            # expect {"status":"ok", ...}
```

`status` reports `degraded` when the database is unreachable or config is
incomplete — check `config_problems`.

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env          # set DB_PASSWORD, ALLOW_HEADER_AUTH=true, APP_ENV=development
python scripts/init_db.py
python scripts/seed_faq.py
uvicorn apps.api.main:app --reload
```

With `ALLOW_HEADER_AUTH=true` you can pass `X-User-Id: u1` instead of a JWT.
`Settings.validate()` refuses to call a production config healthy while that is
on — it lets any caller claim any user id.

```bash
python -m apps.worker.worker              # worker, foreground
python scripts/enqueue_job.py --list      # registered job types
```

## Layout

```
apps/
  api/          FastAPI: the single endpoint, auth, rate limits, action handlers
  worker/       job consumer (cron --once, or daemon)
packages/
  config/       env-driven settings
  domain/       Pydantic contract shared by everything
  common/       question normalisation used by cache, FAQ and mining alike
  storage/      engine, migrations, vectors, repositories (the only SQL)
  cache/        L1 LRU + L2 exact + L3 semantic, and the cache key
  jobs/         handler registry
  orchestrator/ graph nodes, router cascade, response templates
scripts/        init_db, seed_faq, enqueue_job
```

## What is built, and what is not

Built and tested end to end: the data plane, consent scoping, the three-tier
cache with the arch.md §7.2 key, the router cascade stages S0–S2, deterministic
template rendering, the job queue and worker, tracing, and the single endpoint
with auth and rate limits.

Also built: the full scan pipeline — fetch broker, blob store, barcode decode
with check-digit validation and multi-frame voting, cached OCR, product
identification cascade, panel parsing, the chemical resolver ladder, the hazard
rules engine, personal risk matching, and the `guard_out` claim verification.

Not yet wired — these return honest "not connected yet" results rather than
fabricated ones:

| Gap | arch.md phase |
|---|---|
| LLM chains: S4 classifier, embeddings, synthesis, memory summarisation | 2 |
| Embedding half of hybrid FAQ retrieval + cross-encoder rerank | 3 |
| Chemical KB **ETL** (the engine and schema exist; bulk ingest does not) | 4 |
| LLM explanation layer over the analyzer's findings | 5 |
| Evidence service with independence scoring | 6 |
| Restaurant analyzer | 7 |

The KB currently holds a 12-chemical starter set. Coverage of the top ~5,000
INCI ingredients is what makes "unresolved ingredient" rare in practice, and
that is an ETL job, not more code.

The tables, job types and response blocks for all of them exist, so each is a
matter of filling in a handler rather than reshaping the system.

## Migration status

`app.py` — the Flask monolith this architecture replaces — has been removed from
this repo (arch.md §15 phase 8). The copy deployed on the server is untouched
and remains the rollback until traffic has moved.

Everything it did is covered:

| Old behaviour | Now |
|---|---|
| barcode retail-symbology gate | `packages/product/barcode.py`, plus check digits, GTIN normalisation, frame voting |
| OCR (run twice per image) | `packages/product/ocr.py`, once, cached on image hash |
| 6-API-per-ingredient fan-out | Chemical KB lookup; the APIs moved to `packages/etl/` |
| `csv_health_data` in the payload | `sync` action → MySQL → user-scoped tools |
| file cache under `database/` | three-tier cache with the arch.md §7.2 key |
| tool-calling chat loop | `packages/chains/personal.py` with 8 scoped tools |
| `urlopen` on user URLs | `packages/guards/fetch_broker.py` |
| prompt-directive control flow | typed `product_unidentified` blocks |
"# Backend-Architecture" 
