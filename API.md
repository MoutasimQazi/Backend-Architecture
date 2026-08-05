# API guide for the frontend

Everything is **one endpoint**. You POST JSON (or multipart when sending
images) to `/`, and an `action` field decides what happens.

```
Base URL   https://fitness.moveneticsdigital.com/
Method     POST /          everything
           GET  /          unauthenticated status/health
```

---

## 1. Authentication

Send a bearer JWT on every request.

```http
POST / HTTP/1.1
Authorization: Bearer <jwt>
Content-Type: application/json
```

The token is HS256, signed with the server's `JWT_SECRET`, and must contain:

| claim | value |
|---|---|
| `sub` | your user id (any stable string) |
| `aud` | `fitness-api` |
| `exp` | expiry |

```js
// server-side only - never ship JWT_SECRET to the client
jwt.sign({ sub: userId, aud: "fitness-api" }, SECRET, { expiresIn: "1h" });
```

Mint tokens on **your** backend. If the mobile app signs its own tokens, the
secret is in the binary and anyone can impersonate any user.

`401` means the token is missing, malformed or expired. The response never says
which — that would tell an attacker what to fix.

---

## 2. Response envelope

Every response, success or failure, has the same outer shape.

```jsonc
// 200
{ "success": true, "request_id": "8885a099c83f4a67", "action": "chat", /* ...action fields */ }

// 4xx / 5xx
{ "success": false, "error": "human-readable reason", "request_id": "..." }
```

`X-Request-Id` is also a response header. **Log it.** It is the only way to
correlate a user's complaint with a server-side trace.

| status | meaning |
|---|---|
| 400 | bad body, unknown action, too many attachments, base64 attachment |
| 401 | auth failed |
| 413 | body or file over the limit |
| 429 | rate limited |
| 500 | server error — `error` is always the literal string `"internal error"` |

Rate limits default to **30/min and 500/day per user**. The limiter fails open:
if it cannot reach the database the request is allowed, so a limiter outage
never takes the app down.

---

## 3. Health check

```http
GET /
```

No auth. Use it for your status page and to verify a deploy.

```jsonc
{
  "success": true,
  "status": "ok",                  // or "degraded"
  "env": "production",
  "database": { "connected": true, "server": "10.11.16-MariaDB", "name": "..." },
  "versions": { "prompt": "v1", "kb": "v1" },
  "providers": { "openai": true, "deepseek": true, "huggingface": true },
  "actions": ["chat", "scan", "upload", /* ... */],
  "config_problems": 0
}
```

`status: "degraded"` means the DB is unreachable **or** the config has problems.
Treat anything other than `"ok"` as not ready.

---

## 4. Actions

### `chat` — the main conversational turn

```jsonc
// request
{
  "action": "chat",
  "message": "how much protein should I eat?",
  "session_id": "optional; omit to start a new one",
  "locale": "en",
  "jurisdiction": "IN"
}
```

```jsonc
// response
{
  "success": true,
  "action": "chat",
  "turn_id": "3ad48bb6...",
  "session_id": "10762b7f...",     // KEEP THIS and send it on the next turn
  "message": "plain text, all blocks joined - use as a fallback",
  "payload": {
    "blocks": [ /* see §6 - render these */ ],
    "citations": [],
    "disclaimers": [],
    "confidence": 0.82,
    "confidence_reason": null,
    "data_gaps": []
  },
  "route": { "label": "FAQ", "confidence": 0.95, "stage": "S1_EXACT" },
  "cache": "MISS",
  "pending_jobs": [],
  "latency_ms": 412
}
```

**Render `payload.blocks`, not `message`.** `message` is a flattened fallback
for notifications and accessibility. The blocks carry the structure.

Persist `session_id` for the conversation. Without it every turn starts fresh
and the assistant loses context.

`pending_jobs` is non-empty when background work was started (see §5).

---

### `scan` — product scan (images)

**Multipart only.** Base64 is rejected.

```http
POST / HTTP/1.1
Authorization: Bearer <jwt>
Content-Type: multipart/form-data; boundary=...

action=scan
message=can I eat this?
file=<binary jpeg/png>
```

```js
const fd = new FormData();
fd.append("action", "scan");
fd.append("message", "can I eat this?");
fd.append("file", photoBlob, "label.jpg");     // repeat "file" for several frames

await fetch(BASE, { method: "POST", headers: { Authorization: `Bearer ${jwt}` }, body: fd });
```

- File field: `file` (also accepted: `files`, `image`, `images`, `attachment`,
  `attachments`). **Repeat the field** for multiple frames — several angles of
  the same pack improve barcode confidence through multi-frame voting.
- Max **8 MB per file**, **5 files**.
- Accepted: `image/jpeg`, `image/png`, `image/webp`, `image/heic`,
  `application/pdf`.
- Structured params: send them as ordinary form fields, or as one `payload`
  field containing a JSON object.
- Sending files with **no** `action` defaults to `scan`.

Response is the same shape as `chat`, with product-specific blocks.

> **Scans take 20–30 seconds.** Barcode decode, OCR, resolution, rules and an
> LLM explanation. Set your client timeout to at least 60 s and show real
> progress, not a spinner that looks hung.

---

### `upload` — store images, get handles

Use this when you want to upload once and analyse later, or retry without
re-sending bytes.

```js
const fd = new FormData();
fd.append("action", "upload");
fd.append("file", blob, "label.jpg");
```

```jsonc
{
  "action": "upload",
  "attachments": [{
    "attachment_id": "5095aa07...",
    "mime_type": "image/jpeg",
    "size_bytes": 137783,
    "sha256": "661b173e...",
    "deduplicated": false
  }],
  "errors": []
}
```

Then reference the handle in a later turn:

```jsonc
{ "action": "scan", "message": "is this safe?",
  "attachments": [{ "attachment_id": "5095aa07...", "mime_type": "image/jpeg" }] }
```

`deduplicated: true` means those exact bytes were already stored for this user —
a free re-upload, not an error.

One bad file does not sink the others: it appears in `errors` while the rest
still store.

---

### `sync` — push health data

The app writes health data here instead of shipping the dataset on every turn.

```jsonc
{
  "action": "sync",
  "profile":  { "display_name": "...", "date_of_birth": "1994-03-02", "sex": "female",
                "height_cm": 168, "weight_kg": 61.5, "pregnancy_status": "none" },
  "metrics":  [{ "metric": "weight", "measured_on": "2026-08-01", "value": 61.5,
                 "unit": "kg", "source": "manual" }],
  "nutrition":[{ "consumed_on": "2026-08-01", "calories": 2100, "protein_g": 130 }],
  "activities":[{ "started_at": "2026-08-01T06:30:00Z", "activity_type": "run",
                  "duration_min": 42 }],
  "medical":  { "report_date": "2026-07-20", "conditions": ["PCOS"],
                "allergies": ["peanuts"], "medications": ["metformin"] }
}
```

```jsonc
{ "action": "sync", "written": { "profile": 1, "metrics": 1 },
  "aggregate_job_id": "..." }
```

Idempotent — metrics upsert on `(user, metric, date, source)`. Safe to re-send.

`source` matters: a value from a wearable and one the user typed are both kept.

---

### `consent` — required before health data is used

Call this **first**, during onboarding, before `sync`.

```jsonc
{ "action": "consent", "op": "grant", "scopes": ["profile", "vitals"] }
// op: "list" | "grant" | "revoke"
```

```jsonc
{ "action": "consent",
  "granted": ["profile", "vitals"],
  "available": ["vitals", "nutrition", "labs", "activity", "location", "profile"] }
```

| scope | unlocks |
|---|---|
| `profile` | name, age, sex, height, weight, **and allergies/conditions/medications** |
| `vitals` | weight, sleep, heart rate and other metrics |
| `nutrition` | food logs |
| `activity` | workouts |
| `labs` | lab values in the medical section |
| `location` | location-based features |

Without the relevant scope, that section is withheld from the assistant and it
will say so rather than guessing.

> Revoking consent invalidates the user's cached answers, because what the
> assistant was allowed to see has changed.

---

### `context` — what the assistant knows

Use it to render a "your data" screen or debug why an answer was generic.

```jsonc
{
  "action": "context",
  "profile": { /* ... */ },
  "profile_version": 4,
  "consent": ["profile", "vitals"],
  "aggregate_versions": { "vitals": "12", "nutrition": "8" },
  "latest_metrics": [{ "metric": "weight", "value": 61.5, "unit": "kg",
                       "measured_on": "2026-08-01" }]
}
```

---

### `history` — recent turns

```jsonc
{ "action": "history", "session_id": "...", "limit": 20 }   // session_id REQUIRED
```

```jsonc
{ "action": "history",
  "turns": [{ "role": "user", "content": "...", "created_at": "2026-08-05 12:01:33" }] }
```

---

### `memory` — durable facts, user-visible and user-deletable

```jsonc
{ "action": "memory", "op": "list" }                                 // or omit op
{ "action": "memory", "op": "list", "kind": "goal" }                 // filter
{ "action": "memory", "op": "remember", "kind": "goal", "value": "run a 10k" }
{ "action": "memory", "op": "forget", "memory_id": 42 }
```

`op` is one of `list` | `remember` | `forget` — anything else is a 400.

```jsonc
{ "action": "memory", "memories": [ /* ... */ ] }
{ "action": "memory", "stored": true }
{ "action": "memory", "forgotten": 1 }
```

Kinds: `goal`, `dietary_restriction`, `disliked_exercise`, `preferred_exercise`,
`constraint`, `injury`, `schedule`, `equipment`, `motivation`.

**Build a screen for this.** The assistant captures facts from conversation
automatically; users need to see and delete them. That is both a trust and a
privacy requirement.

---

### `job_status` / `job_enqueue`

```jsonc
{ "action": "job_status", "job_id": "abc..." }     // one job
{ "action": "job_status" }                          // this user's recent jobs
```

```jsonc
{ "action": "job_status",
  "job": { "job_id": "...", "job_type": "chemical_research",
           "status": "succeeded", "attempts": 1, "result": { /* ... */ } } }
```

`status`: `queued` | `running` | `succeeded` | `failed`.

---

### `admin.*`

`admin.cache`, `admin.faq`, `admin.jobs`. Require an admin token. Not for the
client app.

---

## 5. Background work — `pending_jobs`

Some answers start work that finishes later. The turn returns immediately with a
`job_pending` block and a job id.

```
scan a product with unknown ingredients
  -> answer now, with what is known
  -> pending_jobs: ["job-abc"]
  -> research runs in the background
  -> a later scan of the same product knows more
```

Recommended handling: show the `job_pending` block text, and **do not poll
aggressively**. Chemical research takes 10–60 s. Poll `job_status` every ~10 s
for up to a minute, or simply let the next scan pick up the improvement.

---

## 6. Rendering `payload.blocks`

Each block is `{ block_id, type, text, data }`. Render by `type`; fall back to
`text` for any type you do not handle yet, so a new block type never breaks the
UI.

| type | contains | render as |
|---|---|---|
| `text` | `text` | paragraph |
| `faq_answer` | `text`, `data.faq_id`, `data.version` | paragraph |
| `hazard_badge` | `data.verdict` | **coloured badge — the headline** |
| `ingredient_table` | `data.rows[]` | table (see below) |
| `metric_card` | `data.metrics[]`, `data.trends` | cards / chart |
| `evidence_list` | `data.sources[]` | citation list |
| `product_unidentified` | `data.reason` | retry prompt |
| `job_pending` | `data.job_ids` | "working on it" note |
| `safety_notice` | `text` | prominent notice |
| `action_prompt` | `data` | call-to-action button |

### Verdicts — exactly four values

| `data.verdict` | suggested colour |
|---|---|
| `"Generally suitable"` | green |
| `"Use with caution"` | amber |
| `"Not recommended for you"` | red |
| `"Insufficient data"` | grey |

These are a **fixed enum produced by the rules engine**, never by the model.
You can switch on them safely.

> `"Insufficient data"` is not an error. It means too little of the panel could
> be assessed to make a claim. Render it as grey/neutral, never as "safe".

### `ingredient_table` rows

```jsonc
{
  "position": 10,
  "name": "hydrolysed groundnut protein",
  "raw": "hydrolysed groundnut protein",
  "recognised": false,
  "resolution": "unresolved",
  "confidence": 0.0,
  "hazard_level": "unknown",          // none | low | moderate | high | unknown
  "iarc_group": null,
  "endocrine": false,
  "allergen": false,
  "banned_in": [],
  "restricted_in": [],
  "caveat": null,
  "rules_fired": [],
  "personal_flags": ["contains groundnut, which is the same allergen as your declared peanuts"]
}
```

**`personal_flags` is the most important field on the screen.** It is
user-specific and safety-critical. Sort flagged rows to the top and make them
visually unmissable.

Distinguish `hazard_level: "unknown"` (nobody has assessed it) from `"none"`
(assessed, nothing found). Do not render "unknown" as safe.

---

## 7. Two flows to implement

### Onboarding

```
1. mint JWT on your backend
2. POST { action: "consent", op: "grant", scopes: [...] }   <- before any data
3. POST { action: "sync", profile: {...}, medical: {...} }
4. POST { action: "chat", message: "..." }                  <- keep session_id
```

Consent first. `sync` before consent stores data the assistant is not permitted
to read.

### Scan

```
1. capture 1-3 frames of the pack (barcode side + ingredients panel)
2. multipart POST: action=scan + file (repeated)
3. render hazard_badge -> personal_flags -> ingredient_table
4. if pending_jobs, show the job_pending note
```

Prompt the user to include **both** the barcode and the ingredients panel. The
barcode identifies the product; the panel is the fallback when the product is
not in any database — and it is what makes an unlisted product still work.

---

## 8. Things that will bite you

| | |
|---|---|
| **Don't send base64.** | Rejected with a 400 telling you to use multipart. It inflated every image by a third and hit the JSON body limit. |
| **Don't discard `session_id`.** | Each turn becomes a cold start. |
| **Don't render `message` instead of blocks.** | You lose the verdict badge, the table and the flags. |
| **Don't treat `"Insufficient data"` as safe.** | It means unknown. |
| **Don't poll `job_status` in a tight loop.** | Rate limit is 30/min. |
| **Don't sign JWTs in the app.** | The secret ends up in the binary. |
| **Set a 60 s timeout for scans.** | 20–30 s is normal. |
| **Log `request_id`.** | It is how anything gets diagnosed. |

---

## 9. Quick reference

```
POST /   Authorization: Bearer <jwt>

action           purpose                          content-type
---------------  -------------------------------  -----------------
chat             conversational turn              json
scan             analyse product images           multipart
upload           store images, get handles        multipart
sync             push health data                 json
consent          grant/revoke/list scopes         json
context          what the assistant knows         json
history          recent turns                     json
memory           list/store/forget facts          json
job_status       background job state             json
job_enqueue      start a job                      json
admin.cache      admin                            json
admin.faq        admin                            json
admin.jobs       admin                            json
```

Limits: 30 req/min, 500/day, 8 MB/file, 5 files, 2 MB JSON body.
