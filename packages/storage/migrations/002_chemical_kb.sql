-- ---------------------------------------------------------------------------
-- 002_chemical_kb — the curated store that moves ingredient research offline
--
-- arch.md 8.4: a scan at runtime touches zero external toxicology APIs. These
-- tables are filled by the ETL, reviewed, and versioned; the request path only
-- reads them.
-- ---------------------------------------------------------------------------

SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS chemical (
  chemical_id  VARCHAR(64)  NOT NULL,
  inci_name    VARCHAR(255) NULL,
  display_name VARCHAR(255) NOT NULL,
  cas          VARCHAR(32)  NULL,
  ec           VARCHAR(32)  NULL,
  e_number     VARCHAR(16)  NULL,
  formula      VARCHAR(128) NULL,
  chem_class   VARCHAR(128) NULL,
  functions    JSON         NULL,
  kb_version   VARCHAR(32)  NOT NULL DEFAULT 'v1',
  review_status VARCHAR(16) NOT NULL DEFAULT 'draft', -- draft | reviewed | published
  reviewed_by  VARCHAR(191) NULL,
  reviewed_at  DATETIME(3)  NULL,
  created_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (chemical_id),
  UNIQUE KEY uq_chem_inci (inci_name),
  KEY ix_chem_cas (cas),
  KEY ix_chem_ec (ec),
  KEY ix_chem_enum (e_number),
  KEY ix_chem_published (review_status, kb_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 8.3 step 3: the resolution ladder reads this table for the
-- synonym and embedding stages. `norm_text` is what the fuzzy matcher sees.
CREATE TABLE IF NOT EXISTS chemical_synonym (
  id           BIGINT       NOT NULL AUTO_INCREMENT,
  chemical_id  VARCHAR(64)  NOT NULL,
  synonym      VARCHAR(255) NOT NULL,
  norm_text    VARCHAR(255) NOT NULL,
  norm_hash    CHAR(64)     NOT NULL,
  kind         VARCHAR(24)  NOT NULL DEFAULT 'synonym', -- inci | cas | ec | e_number | trade | synonym
  locale       VARCHAR(16)  NULL,
  embedding    VARBINARY(8192) NULL,
  embedding_model VARCHAR(64) NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_syn (norm_hash, chemical_id),
  KEY ix_syn_hash (norm_hash),
  KEY ix_syn_chem (chemical_id),
  FULLTEXT KEY ft_syn_text (synonym),
  CONSTRAINT fk_syn_chem FOREIGN KEY (chemical_id) REFERENCES chemical (chemical_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per hazard/regulatory assertion, each carrying its own provenance
-- (arch.md 8.4 `provenance` is per-field, not per-dossier).
CREATE TABLE IF NOT EXISTS chemical_assertion (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  chemical_id   VARCHAR(64)  NOT NULL,
  domain        VARCHAR(32)  NOT NULL,  -- hazard | regulatory | endocrine | allergen | absorption
  key_name      VARCHAR(64)  NOT NULL,  -- ghs_code | iarc_group | svhc | annex_ii | prop65 | ...
  value         VARCHAR(255) NULL,
  jurisdiction  VARCHAR(16)  NULL,      -- EU | US | IN | INTL
  product_class VARCHAR(24)  NULL,      -- cosmetic | food | drug  (arch.md 8.5)
  limit_value   DECIMAL(12,6) NULL,
  limit_unit    VARCHAR(32)  NULL,
  evidence_grade VARCHAR(16) NULL,      -- A | B | C | D
  source        VARCHAR(128) NULL,      -- ECHA | IARC | CosIng | FDA | FSSAI | PubChem
  source_url    VARCHAR(512) NULL,
  fetched_at    DATETIME(3)  NULL,
  reviewed_by   VARCHAR(191) NULL,
  kb_version    VARCHAR(32)  NOT NULL DEFAULT 'v1',
  PRIMARY KEY (id),
  KEY ix_assert_lookup (chemical_id, domain, jurisdiction, product_class),
  KEY ix_assert_key (key_name, value),
  CONSTRAINT fk_assert_chem FOREIGN KEY (chemical_id) REFERENCES chemical (chemical_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 9: evidence documents, tiered and independence-scored rather than
-- filtered on "declares a grant" (which inverted the intended signal, P6).
CREATE TABLE IF NOT EXISTS evidence_document (
  source_id     VARCHAR(96)  NOT NULL,  -- doi:… | pmid:… | url-hash
  title         TEXT         NULL,
  container     VARCHAR(255) NULL,
  url           VARCHAR(768) NULL,
  tier          VARCHAR(8)   NOT NULL DEFAULT 'T3',
  published_year SMALLINT    NULL,
  study_design  VARCHAR(64)  NULL,
  funder_class  VARCHAR(32)  NULL,      -- public | charitable | industry | none | unknown
  declared_coi  TINYINT(1)   NULL,
  sponsor_role  VARCHAR(64)  NULL,
  registry_status VARCHAR(32) NULL,
  independence  DECIMAL(4,3) NULL,      -- shown to the user, never used to hide
  abstract      MEDIUMTEXT   NULL,
  retrieved_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (source_id),
  KEY ix_evidence_rank (tier, independence DESC, published_year DESC),
  FULLTEXT KEY ft_evidence (title, abstract)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evidence_chunk (
  id          BIGINT      NOT NULL AUTO_INCREMENT,
  source_id   VARCHAR(96) NOT NULL,
  chunk_index INT         NOT NULL DEFAULT 0,
  chunk_text  MEDIUMTEXT  NOT NULL,
  embedding   VARBINARY(8192) NULL,
  embedding_model VARCHAR(64) NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_chunk (source_id, chunk_index),
  FULLTEXT KEY ft_chunk (chunk_text),
  CONSTRAINT fk_chunk_doc FOREIGN KEY (source_id) REFERENCES evidence_document (source_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chemical_evidence (
  chemical_id VARCHAR(64) NOT NULL,
  source_id   VARCHAR(96) NOT NULL,
  relation    VARCHAR(32) NOT NULL DEFAULT 'general', -- absorption | carcinogenicity | endocrine | general
  note        VARCHAR(512) NULL,
  PRIMARY KEY (chemical_id, source_id, relation),
  KEY ix_chemev_source (source_id),
  CONSTRAINT fk_chemev_chem FOREIGN KEY (chemical_id) REFERENCES chemical (chemical_id) ON DELETE CASCADE,
  CONSTRAINT fk_chemev_doc FOREIGN KEY (source_id) REFERENCES evidence_document (source_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 8.5: rules are declarative and versioned so a verdict is reproducible.
CREATE TABLE IF NOT EXISTS hazard_rule (
  rule_id       VARCHAR(64)  NOT NULL,
  version       INT          NOT NULL DEFAULT 1,
  active        TINYINT(1)   NOT NULL DEFAULT 1,
  product_class VARCHAR(24)  NULL,
  jurisdiction  VARCHAR(16)  NULL,
  priority      INT          NOT NULL DEFAULT 100,
  condition_json JSON        NOT NULL,  -- matched against chemical_assertion rows
  effect_json   JSON         NOT NULL,  -- hazard_level, flags, caveat text id
  description   VARCHAR(512) NULL,
  owner         VARCHAR(191) NULL,
  created_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (rule_id, version),
  KEY ix_rule_active (active, product_class, jurisdiction, priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 8.6: allergy cross-reactant expansion is data, not a prompt.
CREATE TABLE IF NOT EXISTS allergen_cross_reactant (
  allergen_key VARCHAR(96) NOT NULL,
  chemical_id  VARCHAR(64) NOT NULL,
  severity     VARCHAR(16) NOT NULL DEFAULT 'moderate',
  source       VARCHAR(128) NULL,
  PRIMARY KEY (allergen_key, chemical_id),
  KEY ix_xreact_chem (chemical_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Product identification cascade results (arch.md 8.2), cached locally so the
-- second scan of the same barcode never leaves the box.
CREATE TABLE IF NOT EXISTS product (
  product_id     VARCHAR(64)  NOT NULL,
  barcode        VARCHAR(32)  NULL,
  barcode_format VARCHAR(32)  NULL,
  name           VARCHAR(255) NULL,
  brand          VARCHAR(191) NULL,
  category       VARCHAR(128) NULL,
  product_class  VARCHAR(24)  NULL,      -- cosmetic | food | drug
  ingredients_text MEDIUMTEXT NULL,
  parsed_ingredients JSON     NULL,
  source         VARCHAR(64)  NULL,      -- off | obf | opf | gs1 | user | manual
  confidence     DECIMAL(4,3) NOT NULL DEFAULT 0.000,
  fetched_at     DATETIME(3)  NULL,
  expires_at     DATETIME(3)  NULL,
  created_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (product_id),
  KEY ix_product_barcode (barcode),
  KEY ix_product_expiry (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- arch.md 9.2: the allowlist is a table with an owner and a review date.
CREATE TABLE IF NOT EXISTS source_allowlist (
  domain       VARCHAR(191) NOT NULL,
  tier         VARCHAR(8)   NOT NULL DEFAULT 'T4',
  allowed      TINYINT(1)   NOT NULL DEFAULT 1,
  owner        VARCHAR(191) NULL,
  review_due   DATE         NULL,
  note         VARCHAR(512) NULL,
  PRIMARY KEY (domain),
  KEY ix_allowlist_tier (tier, allowed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
