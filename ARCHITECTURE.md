# Architecture

A health and product assistant: predefined Q&A, personalised answers from the
user's own health data, evidence-backed research, and a product/ingredient
scanner that reads barcodes and label photos.

This document describes what is actually built and running, including the
places where it deliberately differs from the original design.

---

## 1. Shape of the system

```
                         POST /   (one endpoint, "action" discriminates)
                                        |
                            auth -> rate limit -> action handler
                                        |
                                 run_turn(state)
                                        |
  ingest -> guard_in -> context_build -> router -> cache_probe -> BRANCH
                                                        |
                            compose -> guard_out -> persist -> (background jobs)
```

Every conversational turn is one `ConversationState` object threaded through a
fixed sequence of nodes, each timed. `packages/orchestrator/pipeline.py`.

**One endpoint by design.** cPanel/Passenger routing stays trivial, and the
existing client keeps a single URL. Thirteen actions live behind it.

### Not literally LangGraph

The original design drew this as a LangGraph `StateGraph`. It is a for-loop.
The edges are the same and `nodes.py` already has the signature a graph node
needs, so swapping LangGraph in is a change to `pipeline.py` alone. It was left
out because LangGraph's value here — checkpointing and streaming — is not used
by any node yet, and it is a heavy dependency on shared hosting.

---

## 2. The router cascade

The single most important cost decision. Five stages; **the first four are
free**, and a message only reaches a paid model if the free stages cannot
settle it.

| Stage | What it does | Cost |
|---|---|---|
| **S0_RULES** | Deterministic. Attachments present → PRODUCT. Greeting regex → SMALLTALK. Safety triggers → UNSAFE. | $0 |
| **S1_EXACT** | Exact hit on a normalised question hash, including every stored paraphrase. | $0 |
| **S2_RETRIEVAL** | Lexical FAQ retrieval (InnoDB FULLTEXT) above a per-category threshold. | $0 |
| **S3_SEMANTIC_CACHE** | Has anyone asked this before? | $0 |
| **S4_LLM** | Small-model classifier. Only the genuinely ambiguous band. | ~$0.00002 |

Measured: smalltalk, FAQ and repeat questions cost **$0.000000**. A full
product scan costs **$0.000425**.

> The enum value is still `"S2_EMBEDDING"` for trace-compatibility, but the
> constant is `RouteStage.S2_RETRIEVAL` — with embeddings off it is a pure
> lexical match, and a stage labelled "embedding" sent people looking for a
> vector search that never ran.

### Branches

`SMALLTALK` → template · `FAQ`/`CACHED` → slot-filled template ·
`PERSONAL` → tool-calling agent over the user's health data ·
`RESEARCH` → evidence pipeline · `PRODUCT` → scan pipeline ·
`RESTAURANT` → analyzer (framework only, see §10) ·
`UNSAFE` → **deterministic safety template, never a model**.

---

## 3. Product scan pipeline

```
image bytes (multipart)
   |
   +- barcode: zxing -> retail-symbology gate -> check digit -> GTIN-14
   |                 -> multi-frame voting
   |     found? -> local product table -> OpenFoodFacts -> ingredients_text
   |
   +- OCR: sha256-cached -> ingredient panel extraction
                            (header -> next section heading)
   |
   v
parser: paren-depth splitting, compound expansion, qualifiers, traces
   |
   v
resolver ladder:
   exact hash -> CAS/EC/E-number -> qualifiers -> OCR-confusion variants
   -> edit distance  [-> embedding NN, disabled]
   |
   v
hazard rules engine (declarative, 9 default rules)
   |
   v
personal matcher: allergies (synonym-aware), cross-reactants, conditions,
                  pregnancy, diet
   |
   v
verdict (4-value enum)  ->  LLM writes prose ABOUT findings  ->  guard_out
```

Verified end to end on real product photos: a UPC-A candy pack (17/17
ingredients read from an angled, glare-affected photo) and an EAN-13 Indian
barcode (product identified, ingredients from the product database).

### Compound ingredient expansion

A bracketed list containing a comma is a **sub-ingredient list**, not a
qualifier, and each part becomes its own token — recursively, because they
nest:

```
Masala Tastemaker (hydrolysed groundnut protein, mixed spices, Noodle powder,
                   ..., Thickeners (508 & 412), ...)
```

This matters because **that bracket is where labels declare allergens**.
Keeping only the outer name deleted `hydrolysed groundnut protein` entirely,
so a declared peanut allergy could not fire no matter how good the matching
was. It also exposes EU-26 fragrance allergens declared inside `Parfum (...)`.

### Allergen synonym expansion

`packages/product/allergens.py`. Twelve families, built for the Indian market:
groundnut/moongphali (peanut), kaju (cashew), badam (almond), maida/atta/suji
(wheat), til/gingelly (sesame), jhinga (prawn), sarson (mustard),
paneer/ghee/khoya (milk).

Two properties it must have:

1. **Expansion is additive.** The declared term always matches too. Nothing
   here can make an allergen stop matching.
2. **Exclusions prevent false alarms, not real ones.** Cocoa butter and shea
   butter are not dairy; eggplant is not egg; buckwheat is not wheat. An alert
   on everything teaches people to ignore the one that matters.

It matches **raw label text**, so it fires on ingredients the Chemical KB has
never heard of — which is the common case.

---

## 4. Chemical Knowledge Base and the self-healing loop

An unknown ingredient does not block the scan. It enqueues a research job and
the response says so:

```
scan -> unresolved token -> CHEMICAL_RESEARCH job -> PubChem/EuropePMC
     -> dossier written as `draft` -> next scan resolves it
```

Measured across three scans of the same photo: **0/17 → 7/17 → 11/17**
recognised. The KB grows by itself.

Two things make that loop work:

- **Research is prioritised, not first-come.** Panels are ordered by weight, so
  the first ten unresolved tokens are bulk (sugar, corn syrup, starch) and the
  colours sit at positions 12–17. Taking `unresolved[:10]` spent the entire
  research budget on sugar while Red #40 and Tartrazine — the ingredients
  carrying mandatory EU warnings — were never looked up. Tokens are now ranked
  by how likely they are to carry hazard data.
- **Label names are aliased to chemical names.** PubChem 404s on `red #40` and
  resolves `Allura Red AC` (CID 33258). Both converge on one dossier, and the
  label wording is stored as a synonym so the next scan resolves on the first
  rung without another API call.

**Nothing is auto-published.** A dossier assembled from a public API lands as
`draft`; a reviewer promotes it. Hazard rules fire on assertions, and an
unreviewed assertion is visibly attributed to its source.

### Draft vs reviewed

- Dossier with assertions → those assertions produce findings.
- Dossier **reviewed** with no assertions → `NONE`. A reviewer looked and found
  nothing; that is a finding.
- Dossier **draft** with no assertions → `UNKNOWN`. Nobody has looked yet.
  Absence of data is never reported as safety.

### In-process cache

`packages/product/chem_cache.py`. Keyed on a version token derived from the KB's
own state (`kb_version : count : max(updated_at)`), so an ETL write makes stale
entries unreachable rather than merely old — which matters more here than in an
ordinary cache, because the output is a safety verdict.

```
cold  (first scan)   3 SQL queries   387 ms
warm                 0 queries         0 ms
negative caching     unknown ingredient cached as absent
writes               invalidate immediately
```

Misses are cached too: an ingredient the KB has never heard of is the most
repeated lookup in the system.

---

## 5. Safety guarantees

These are the properties the design exists to hold. Each is enforced in code,
not by prompt instruction.

| Guarantee | Mechanism |
|---|---|
| The model never decides a hazard level | Rules engine assigns levels; the LLM only writes prose |
| The verdict is from a fixed set | 4-value `Verdict` enum, produced by rules |
| The model cannot invent ingredients | `_strip_hallucinated` drops notes naming anything not in the findings |
| The model cannot contradict findings | `guard_out` re-verifies verdict and ingredient names against the structured analysis |
| Absence of data is never safety | Coverage gate → `Insufficient data`; draft dossiers → `UNKNOWN` |
| An allergen match beats a coverage gate | Critical personal flags are checked **before** the coverage gate |
| Allergies are never deleted by inference | `merge_medical` is additive-only; removal is a deliberate user action |
| Allergies are never served stale | Safety fields read live, not from the precomputed aggregate |
| Health data needs consent | Per-scope checks at each write and each context section |
| No SSRF | All outbound HTTP through the fetch broker (§7) |

### Allergy path, specifically

This path had four independent bugs, each alone enough to hide an allergy on a
real product. All four are fixed and pinned by golden cases at a 100% gate:

1. Parser deleted the sub-ingredient list containing the allergen.
2. `groundnut` did not match a declared `peanut`.
3. Flags keyed on `chemical_id` — all unresolved ingredients share `""`, so one
   warning rendered against **12 of 15 rows**.
4. The explainer never received the flags, so it said *"Not recommended for
   you"* and *"no personal flags"* in the same answer.

---

## 6. Personalisation and background capture

The client pushes health data through `sync`. Separately, anything the user
*states in conversation* is captured in the background:

```
chat turn -> cheap regex pre-filter -> PROFILE_CAPTURE job
          -> extraction -> consent check -> health_metric / medical_report / user_memory
```

- **Numbers are normalised in code, not by the model.** `5 foot 10` → 177.8 cm.
  Unconvertible or implausible values are dropped rather than stored, because
  everything downstream treats stored numbers as true.
- **The pre-filter** stops a greeting putting an LLM call behind every turn.
- **Consent is per-destination.** Someone may grant PROFILE but not VITALS:
  "remember my allergy, do not record my weight".

An allergy mentioned in chat therefore reaches the product scanner. That loop
is verified end to end.

### Context assembly

`UserContext` is built from precomputed aggregates (`user_aggregate`), refreshed
by a job, with consent gating each section. Safety-critical fields
(allergies, conditions, medications) are read **live** and unioned in — an
allergy captured two minutes ago is not in yesterday's aggregate, and a stale
allergen list is exactly the failure that hurts someone.

---

## 7. Storage: MariaDB does everything

No Redis, no Docker, no pgvector — shared cPanel hosting.

| Normally | Here |
|---|---|
| Redis cache | `cache_entry` table (L2) + in-process LRU (L1) |
| Redis queue | `job` table, batch claim with a claim token |
| Redis rate limit | `rate_limit_bucket` |
| Redis locks | MySQL `GET_LOCK()` |
| S3 | filesystem + `blob_object` rows |
| pgvector | `VARBINARY(8192)` packed float32 + numpy cosine |
| Postgres FTS | InnoDB `FULLTEXT` + `MATCH…AGAINST` |

31 tables across two migrations.

**Blob dedup is per `(sha256, user_id)`.** Deduping on hash alone handed the
second uploader the first uploader's `blob_id` — a row they do not own — so
every later read returned nothing. The file on disk is still shared.

### Fetch broker

Every outbound HTTP request goes through `packages/guards/fetch_broker.py`:
host allowlist, DNS resolution **before** connecting with every resolved
address checked against private/loopback/link-local ranges, connection pinned
to the checked address (DNS rebinding), manual redirect re-validation, and hard
caps on time, size and redirects. The OCR host is appended from `OCR_API_URL`
so it cannot drift out of the allowlist.

---

## 8. Models and cost

**DeepSeek `deepseek-v4-flash` only.** `deepseek-v4-pro` is *blocked in code*,
not merely unconfigured, so no env var or explicit `model=` can route spend to
it. `deepseek-chat` and `deepseek-reasoner` were retired from the account.

`PROVIDER_ORDER` controls precedence; a key that is set but unlisted stays in
the fallback chain, so a typo degrades rather than causing an outage.

### flash is a reasoning model

It spends completion tokens on hidden reasoning before emitting content. When
the budget runs out mid-thought it returns `finish_reason="length"` with an
**empty string and no error**.

The router was configured at `max_tokens=200`; the measured worst case on real
router prompts is 288. It would have returned nothing on exactly the ambiguous
messages it exists to classify — silently.

Budgets therefore get a multiplier (`max(requested * 3, 2048)`), and an
empty-with-`length` response raises and retries at double. Over-provisioning is
nearly free (you pay for tokens generated); under-provisioning costs a whole
wasted generation. That fix took a scan from 43 s to 23 s.

### Pricing

| | per 1M |
|---|---|
| input, cache miss | $0.14 |
| input, cache hit | $0.0028 |
| output | $0.28 |

Cache hits are priced separately because the gap is 50× and every chain sends a
fixed system prompt ahead of a short user message — hits are the normal case.
Measured: a repeated prompt costs **61% less** than the naive all-miss figure.
Pricing all input at the miss rate overstated spend ~2.6×.

An unpriced model logs a warning rather than silently reporting `$0.00`.

### Permanent errors fail fast

A 400/401/403/404 will return the same answer however often it is asked.
Retrying those three times with backoff turned a config mistake into ~12 s of
dead time *per ingredient*. They now fail immediately and open the breaker;
429 and 5xx still get backoff, which is what backoff is for.

---

## 9. Retrieval is lexical — embeddings are off

`EMBEDDING_PROVIDER=none` by default. Nothing external is called for
embeddings; verified with outbound sockets disabled.

Retrieval was hybrid (FULLTEXT + vectors, fused with Reciprocal Rank Fusion).
The lexical leg works alone — RRF over one list is that list. Embeddings fed
three places: the FAQ second leg, the resolver's **last** rung (after exact,
CAS, synonym, OCR-variant and edit-distance), and evidence chunk search.

Measured with embeddings unavailable: `router 25/26` against an 85% gate,
`faq_retrieval 10/11` against 80%.

What is lost: a paraphrase sharing no words with any FAQ falls through to S4
and costs a fraction of a cent. What is gained: no embedding entitlement to
keep valid, no vector dimension to keep in sync, no third-party outage in the
scan path.

---

## 10. Background jobs

DB-backed queue, no broker. `apps/worker/worker.py --once` for cron; advisory
lock prevents overlap.

| Job | Purpose |
|---|---|
| `CHEMICAL_RESEARCH` | Unknown ingredient → PubChem/EuropePMC → draft dossier |
| `PROFILE_CAPTURE` | Health/preference facts stated in chat → user's record |
| `MEMORY_SUMMARISE` | Rolling conversation summary + structured facts |
| `CONTEXT_AGGREGATE` | Recompute `user_aggregate` sections |
| `DEEP_RESEARCH` | Evidence synthesis for research questions |
| `ETL_CHEMICAL_KB` | Scheduled KB refresh |
| `OCR` | Deferred OCR |
| `RESTAURANT_INVESTIGATION` | Framework only |
| `EMBEDDING_BACKFILL` | No-op while embeddings are off |

---

## 11. Evals

`python scripts/run_evals.py`. Seven suites; the 100% gates cover
deterministic code, so anything below is a bug rather than model variance.

```
router                   PASS  25/26 (96%, gate 85%)
faq_retrieval            PASS  10/11 (91%, gate 80%)
ingredient_parsing       PASS  14/14 (100%, gate 100%)
hazard_rules             PASS  14/14 (100%, gate 100%)
personalisation_safety   PASS  21/21 (100%, gate 100%)
cache_correctness        PASS  11/11 (100%, gate 100%)
independence             PASS  12/12 (100%, gate 100%)
```

One golden case was **corrected** rather than the code: `nested_parenthetical`
expected `CI 77491 (Iron Oxides, Titanium Dioxide)` to collapse to just
`CI 77491`, deleting Titanium Dioxide — IARC group 2B, and one of the hazards
the rules engine exists to catch. The golden set had encoded a bug.

---

## 12. Evidence and sourcing

Sources are tiered and ranked by
`tier × independence × design × recency`.

The original requirement said "government, non-funded sources". A strict
non-funded filter hides most real evidence, so **funder independence is scored**
rather than used as a gate, and industry co-funding outranks public funding in
the negative direction so it cannot be laundered through a public co-sponsor.
Only PubChem and EuropePMC are wired; there is no general web or news search.

---

## 13. Deliberate substitutions and what is not built

| Item | State |
|---|---|
| LangGraph | Sequencer with the same edges; one file to swap |
| pgvector | numpy cosine over `VARBINARY` — and currently off |
| Semantic FAQ / L3 semantic cache | Off with embeddings |
| CSV analytics import | **Not built.** Replaced by the `sync` API |
| Restaurant analyzer | **Framework only.** Place resolution needs a Places provider; all four data stages return "unavailable" rather than a false clean result |
| Food-additive regulatory data | Colours resolve but carry no hazard data — the EU warning is regulatory (Reg. 1333/2008 Annex V), not toxicological, so PubChem does not hold it |

The restaurant stages honestly report "unavailable" because *unavailable* and
*nothing found* mean very different things to someone deciding where to eat.

---

## 14. Layout

```
apps/
  api/          main.py (endpoint), actions.py (13 handlers), security.py
  worker/       worker.py
packages/
  config/       settings.py            env-driven, validate() returns problems
  domain/       enums.py, models.py    typed contract, extra="forbid"
  orchestrator/ pipeline.py, nodes.py, router.py, templates.py
  chains/       providers.py, base.py, classify.py, explain.py, verify.py,
                personal.py, memory.py, capture.py, embeddings.py
  product/      barcode.py, ocr.py, parser.py, resolver.py, rules.py,
                personal.py, allergens.py, analyzer.py, chem_cache.py
  storage/      db.py, blobs.py, vectors.py, migrations/, repositories/
  evidence/     independence.py, tiers.py, synthesis.py
  connectors/   literature.py (PubChem, EuropePMC), openfacts.py
  guards/       fetch_broker.py
  jobs/         registry.py, handlers.py
  evals/        harness.py, suites.py, golden/
  etl/          chemical.py
scripts/        init_db, seed_faq, import_faq, seed_chemicals, run_evals,
                preflight, enqueue_job, check_ddl
```

93 modules, 0 orphans, 0 stubs (one abstract base method, correct).

---

## 15. Deployment

cPanel + Passenger, ASGI→WSGI via `a2wsgi`, app root
`public_html/fitness.moveneticsdigital.com`.

```bash
git pull && pip install -r requirements.txt
touch tmp/restart.txt
```

Required env: `DB_*`, `JWT_SECRET`, `DEEPSEEK_API_KEY`,
`PROVIDER_ORDER=deepseek,openai,huggingface`, `CORS_ORIGINS`, `BLOB_DIR`.

Two standing items:

- **`.htaccess` security rules** live in `deploy/security.htaccess` and are
  appended to the live file (`cat deploy/security.htaccess >> .htaccess`).
  cPanel owns `.htaccess` and injects secrets into it, so git must not manage
  it. Without the rules the source tree is publicly readable.
- **`BLOB_DIR` should sit outside `public_html`**, or uploaded health documents
  are web-reachable.

Cron: `python apps/worker/worker.py --once` every few minutes.
