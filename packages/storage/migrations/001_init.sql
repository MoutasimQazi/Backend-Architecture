-- ---------------------------------------------------------------------------
-- 001_init — base schema
--
-- Target: MySQL 8.0 / MariaDB 10.5+ on cPanel. No CREATE DATABASE here: the
-- cPanel account owns `movenetics_fitness` already and the DB user usually
-- lacks the privilege.
--
-- Deviations from arch.md 3, forced by the host having no Postgres and no Redis:
--   pgvector      -> `embedding` VARBINARY columns (packed float32) scored in
--                    Python. Fine to ~100k rows; swap behind the retriever.
--   PG tsvector   -> InnoDB FULLTEXT indexes (MATCH ... AGAINST).
--   Redis L2      -> `cache_entry`.
--   Redis queue   -> `job` with a claim-token batch claim.
--   Redis limits  -> `rate_limit_bucket`.
--   Redis locks   -> MySQL GET_LOCK()/RELEASE_LOCK(), used by the worker.
--   S3            -> `blob` rows + files under BLOB_DIR.
-- ---------------------------------------------------------------------------

SET NAMES utf8mb4;

-- --------------------------------------------------------------------------
-- Migration bookkeeping
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migration (
  version      VARCHAR(64)  NOT NULL,
  applied_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  checksum     CHAR(64)     NULL,
  PRIMARY KEY (version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Identity, consent
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS app_user (
  user_id        VARCHAR(64)  NOT NULL,
  external_id    VARCHAR(191) NULL,
  email          VARCHAR(191) NULL,
  locale         VARCHAR(16)  NOT NULL DEFAULT 'en',
  jurisdiction   VARCHAR(8)   NOT NULL DEFAULT 'IN',
  status         VARCHAR(16)  NOT NULL DEFAULT 'active',
  created_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (user_id),
  UNIQUE KEY uq_user_external (external_id),
  KEY ix_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 6.3: context_build filters by granted scope and records what it withheld.
CREATE TABLE IF NOT EXISTS consent_scope (
  user_id     VARCHAR(64) NOT NULL,
  scope       VARCHAR(32) NOT NULL,
  granted     TINYINT(1)  NOT NULL DEFAULT 1,
  granted_at  DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  revoked_at  DATETIME(3) NULL,
  PRIMARY KEY (user_id, scope),
  CONSTRAINT fk_consent_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_profile (
  user_id           VARCHAR(64)  NOT NULL,
  display_name      VARCHAR(191) NULL,
  date_of_birth     DATE         NULL,
  age_band          VARCHAR(16)  NULL,
  sex               VARCHAR(16)  NULL,
  height_cm         DECIMAL(5,1) NULL,
  weight_kg         DECIMAL(5,1) NULL,
  pregnancy_status  VARCHAR(32)  NULL,
  goals             JSON         NULL,
  preferences       JSON         NULL,
  version           BIGINT       NOT NULL DEFAULT 1,
  updated_at        DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (user_id),
  CONSTRAINT fk_profile_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Health data plane (arch.md 3: this replaces context.csv_health_data)
-- --------------------------------------------------------------------------

-- Long/narrow so a new wearable metric needs no migration.
CREATE TABLE IF NOT EXISTS health_metric (
  id           BIGINT       NOT NULL AUTO_INCREMENT,
  user_id      VARCHAR(64)  NOT NULL,
  metric       VARCHAR(64)  NOT NULL,   -- recovery | strain | sleep | rhr | weight | ...
  measured_on  DATE         NOT NULL,
  value        DECIMAL(12,4) NULL,
  unit         VARCHAR(32)  NULL,
  source       VARCHAR(64)  NULL,       -- whoop | oura | manual | ...
  raw          JSON         NULL,
  ingested_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_metric_day (user_id, metric, measured_on, source),
  KEY ix_metric_lookup (user_id, metric, measured_on DESC),
  CONSTRAINT fk_metric_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS nutrition_day (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  user_id       VARCHAR(64)  NOT NULL,
  consumed_on   DATE         NOT NULL,
  calories      DECIMAL(10,2) NULL,
  protein_g     DECIMAL(10,2) NULL,
  carbs_g       DECIMAL(10,2) NULL,
  fat_g         DECIMAL(10,2) NULL,
  fiber_g       DECIMAL(10,2) NULL,
  sugar_g       DECIMAL(10,2) NULL,
  sodium_mg     DECIMAL(10,2) NULL,
  water_ml      DECIMAL(10,2) NULL,
  diet_quality  DECIMAL(5,2)  NULL,
  entries       JSON          NULL,
  source        VARCHAR(64)   NULL,
  ingested_at   DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_nutrition_day (user_id, consumed_on),
  KEY ix_nutrition_lookup (user_id, consumed_on DESC),
  CONSTRAINT fk_nutrition_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS activity_session (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  user_id       VARCHAR(64)  NOT NULL,
  activity_type VARCHAR(64)  NOT NULL,
  started_at    DATETIME(3)  NOT NULL,
  duration_min  DECIMAL(8,2) NULL,
  distance_m    DECIMAL(10,2) NULL,
  calories      DECIMAL(10,2) NULL,
  load_score    DECIMAL(8,2) NULL,
  raw           JSON         NULL,
  source        VARCHAR(64)  NULL,
  ingested_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY ix_activity_lookup (user_id, started_at DESC),
  KEY ix_activity_type (user_id, activity_type, started_at DESC),
  CONSTRAINT fk_activity_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Lab values are the most sensitive class: separate table, separate consent scope.
CREATE TABLE IF NOT EXISTS medical_report (
  id           BIGINT       NOT NULL AUTO_INCREMENT,
  user_id      VARCHAR(64)  NOT NULL,
  report_date  DATE         NOT NULL,
  bmi          DECIMAL(6,2) NULL,
  systolic     SMALLINT     NULL,
  diastolic    SMALLINT     NULL,
  hba1c        DECIMAL(5,2) NULL,
  lipids       JSON         NULL,
  labs         JSON         NULL,
  flags        JSON         NULL,
  conditions   JSON         NULL,
  allergies    JSON         NULL,
  medications  JSON         NULL,
  source_blob_id VARCHAR(64) NULL,
  created_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY ix_medical_lookup (user_id, report_date DESC),
  CONSTRAINT fk_medical_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 6.1: aggregates are precomputed by a job, never per request.
-- `version` bumps on every write and is what the cache fingerprint reads.
CREATE TABLE IF NOT EXISTS user_aggregate (
  user_id      VARCHAR(64) NOT NULL,
  section      VARCHAR(32) NOT NULL,   -- vitals | nutrition | activity | medical | derived | profile
  payload      JSON        NOT NULL,
  completeness DECIMAL(4,3) NOT NULL DEFAULT 0.000,
  fresh_as_of  DATETIME(3) NULL,
  version      BIGINT      NOT NULL DEFAULT 1,
  computed_at  DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (user_id, section),
  CONSTRAINT fk_aggregate_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- FAQ knowledge base (arch.md 5.1)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS faq_item (
  id                   VARCHAR(64)  NOT NULL,
  version              INT          NOT NULL DEFAULT 1,
  status               VARCHAR(16)  NOT NULL DEFAULT 'draft',
  category             VARCHAR(32)  NOT NULL DEFAULT 'General',
  canonical_question   TEXT         NOT NULL,
  answer_template      MEDIUMTEXT   NOT NULL,
  variants             JSON         NULL,
  required_slots       JSON         NULL,
  personalisation_rules JSON        NULL,
  safety_class         VARCHAR(32)  NOT NULL DEFAULT 'informational',
  locale               VARCHAR(16)  NOT NULL DEFAULT 'en',
  effective_from       DATETIME(3)  NULL,
  effective_to         DATETIME(3)  NULL,
  owner                VARCHAR(191) NULL,
  reviewed_by          VARCHAR(191) NULL,
  reviewed_at          DATETIME(3)  NULL,
  created_at           DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at           DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY ix_faq_live (status, locale, category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 5.2: each surface form is its own row, so a paraphrase hit is a
-- direct hit rather than an averaged one.
CREATE TABLE IF NOT EXISTS faq_surface (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  faq_id        VARCHAR(64)  NOT NULL,
  surface_text  TEXT         NOT NULL,
  norm_hash     CHAR(64)     NOT NULL,   -- S1 exact stage: O(1), no model
  kind          VARCHAR(16)  NOT NULL DEFAULT 'paraphrase', -- canonical | paraphrase | mined
  locale        VARCHAR(16)  NOT NULL DEFAULT 'en',
  embedding     VARBINARY(8192) NULL,    -- packed float32
  embedding_model VARCHAR(64) NULL,
  embedding_dim SMALLINT     NULL,
  created_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY ix_surface_hash (norm_hash, locale),
  KEY ix_surface_faq (faq_id),
  FULLTEXT KEY ft_surface_text (surface_text),
  CONSTRAINT fk_surface_faq FOREIGN KEY (faq_id) REFERENCES faq_item (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 5.1: unmatched questions cluster into FAQ candidates.
CREATE TABLE IF NOT EXISTS unmatched_question (
  id          BIGINT      NOT NULL AUTO_INCREMENT,
  norm_text   TEXT        NOT NULL,
  norm_hash   CHAR(64)    NOT NULL,
  locale      VARCHAR(16) NOT NULL DEFAULT 'en',
  hits        INT         NOT NULL DEFAULT 1,
  route_label VARCHAR(32) NULL,
  cluster_id  VARCHAR(64) NULL,
  promoted    TINYINT(1)  NOT NULL DEFAULT 0,
  first_seen  DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  last_seen   DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_unmatched (norm_hash, locale),
  KEY ix_unmatched_volume (hits DESC, promoted)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Cache (arch.md 7) — L2 exact + L3 semantic, both here since there is no Redis
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cache_entry (
  cache_key       CHAR(64)     NOT NULL,   -- hash of the arch.md 7.2 tuple
  scope           VARCHAR(64)  NOT NULL DEFAULT 'global',  -- 'global' | user_id
  route_label     VARCHAR(32)  NULL,
  category        VARCHAR(32)  NULL,
  norm_question   TEXT         NULL,
  payload         MEDIUMTEXT   NOT NULL,   -- serialised AnswerPayload
  context_fingerprint VARCHAR(191) NULL,
  prompt_version  VARCHAR(32)  NULL,
  model_id        VARCHAR(64)  NULL,
  kb_version      VARCHAR(32)  NULL,
  locale          VARCHAR(16)  NOT NULL DEFAULT 'en',
  embedding       VARBINARY(8192) NULL,    -- L3 semantic probe
  is_negative     TINYINT(1)   NOT NULL DEFAULT 0,
  hit_count       INT          NOT NULL DEFAULT 0,
  created_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  last_hit_at     DATETIME(3)  NULL,
  expires_at      DATETIME(3)  NOT NULL,
  PRIMARY KEY (cache_key),
  KEY ix_cache_expiry (expires_at),
  KEY ix_cache_scope (scope, route_label, locale, expires_at),
  KEY ix_cache_prune (last_hit_at, hit_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Async job plane (arch.md 2) — DB-backed, claimed by token, no broker
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job (
  job_id           VARCHAR(64)  NOT NULL,
  job_type         VARCHAR(48)  NOT NULL,
  status           VARCHAR(16)  NOT NULL DEFAULT 'queued',
  user_id          VARCHAR(64)  NULL,
  idempotency_key  VARCHAR(191) NULL,
  payload          JSON         NOT NULL,
  result           JSON         NULL,
  error            TEXT         NULL,
  attempts         INT          NOT NULL DEFAULT 0,
  max_attempts     INT          NOT NULL DEFAULT 3,
  priority         INT          NOT NULL DEFAULT 100,
  claim_token      CHAR(32)     NULL,
  worker_id        VARCHAR(64)  NULL,
  available_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  lease_expires_at DATETIME(3)  NULL,
  created_at       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at       DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  finished_at      DATETIME(3)  NULL,
  PRIMARY KEY (job_id),
  UNIQUE KEY uq_job_idempotency (idempotency_key),
  KEY ix_job_claim (status, available_at, priority),
  KEY ix_job_claim_token (claim_token),
  KEY ix_job_lease (status, lease_expires_at),
  KEY ix_job_user (user_id, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Rate limiting (arch.md 13) — fixed-window counters
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rate_limit_bucket (
  subject     VARCHAR(128) NOT NULL,   -- user:<id> | ip:<addr>
  window_key  VARCHAR(32)  NOT NULL,   -- e.g. '2026-08-04T16:41' or '2026-08-04'
  hits        INT          NOT NULL DEFAULT 0,
  expires_at  DATETIME(3)  NOT NULL,
  PRIMARY KEY (subject, window_key),
  KEY ix_rl_expiry (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Blobs (arch.md 3 object store, filesystem-backed here)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS blob_object (
  blob_id     VARCHAR(64)  NOT NULL,
  user_id     VARCHAR(64)  NULL,
  sha256      CHAR(64)     NOT NULL,
  mime_type   VARCHAR(128) NOT NULL,
  size_bytes  BIGINT       NOT NULL,
  rel_path    VARCHAR(255) NOT NULL,
  kind        VARCHAR(32)  NOT NULL DEFAULT 'upload', -- upload | pdf_page | ocr_artifact
  parent_id   VARCHAR(64)  NULL,
  page_index  INT          NULL,
  created_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  expires_at  DATETIME(3)  NULL,
  PRIMARY KEY (blob_id),
  KEY ix_blob_sha (sha256),
  KEY ix_blob_user (user_id, created_at DESC),
  KEY ix_blob_expiry (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- OCR is its own step with its own cache keyed on image hash (arch.md 8.2),
-- which is what kills the current double-OCR.
CREATE TABLE IF NOT EXISTS ocr_result (
  image_sha256 CHAR(64)   NOT NULL,
  lang         VARCHAR(16) NOT NULL DEFAULT 'en',
  ocr_text     MEDIUMTEXT NULL,
  confidence   DECIMAL(5,4) NULL,
  engine       VARCHAR(64) NULL,
  regions      JSON       NULL,
  created_at   DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  expires_at   DATETIME(3) NULL,
  PRIMARY KEY (image_sha256, lang),
  KEY ix_ocr_expiry (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Conversation + memory (arch.md 12)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversation_turn (
  turn_id       VARCHAR(64)  NOT NULL,
  session_id    VARCHAR(64)  NOT NULL,
  user_id       VARCHAR(64)  NOT NULL,
  role          VARCHAR(16)  NOT NULL,
  content       MEDIUMTEXT   NULL,
  payload       JSON         NULL,
  route_label   VARCHAR(32)  NULL,
  route_stage   VARCHAR(32)  NULL,
  route_confidence DECIMAL(5,4) NULL,
  cache_tier    VARCHAR(8)   NULL,
  latency_ms    INT          NULL,
  tokens_in     INT          NOT NULL DEFAULT 0,
  tokens_out    INT          NOT NULL DEFAULT 0,
  cost_usd      DECIMAL(10,6) NOT NULL DEFAULT 0,
  created_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (turn_id),
  KEY ix_turn_session (session_id, created_at),
  KEY ix_turn_user (user_id, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS conversation_summary (
  session_id  VARCHAR(64) NOT NULL,
  user_id     VARCHAR(64) NOT NULL,
  summary     MEDIUMTEXT  NULL,
  turn_count  INT         NOT NULL DEFAULT 0,
  updated_at  DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (session_id),
  KEY ix_summary_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 12: structured memory is queryable and user-editable, not prose.
CREATE TABLE IF NOT EXISTS user_memory (
  id          BIGINT       NOT NULL AUTO_INCREMENT,
  user_id     VARCHAR(64)  NOT NULL,
  kind        VARCHAR(48)  NOT NULL,  -- goal | dietary_restriction | disliked_exercise | ...
  value       VARCHAR(512) NOT NULL,
  confidence  DECIMAL(4,3) NOT NULL DEFAULT 1.000,
  source_turn_id VARCHAR(64) NULL,
  active      TINYINT(1)   NOT NULL DEFAULT 1,
  created_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_memory (user_id, kind, value),
  KEY ix_memory_user (user_id, active),
  CONSTRAINT fk_memory_user FOREIGN KEY (user_id) REFERENCES app_user (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- Observability (arch.md 14) — one trace per turn
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trace_span (
  id          BIGINT       NOT NULL AUTO_INCREMENT,
  turn_id     VARCHAR(64)  NOT NULL,
  node        VARCHAR(64)  NOT NULL,
  started_at  DATETIME(3)  NOT NULL,
  duration_ms DECIMAL(10,2) NOT NULL DEFAULT 0,
  ok          TINYINT(1)   NOT NULL DEFAULT 1,
  error       TEXT         NULL,
  attributes  JSON         NULL,
  PRIMARY KEY (id),
  KEY ix_span_turn (turn_id),
  KEY ix_span_node (node, started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
