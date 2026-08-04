"""Closed vocabularies shared across the system.

These are enums rather than strings on purpose: arch.md 8.6 requires the verdict
to come from a fixed set the guard can check, and arch.md 5.1 keys the router,
TTL policy, disclaimer policy and eval slices off the category taxonomy.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """str-Enum that serialises to its value under Pydantic and JSON."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class RouteLabel(StrEnum):
    SMALLTALK = "SMALLTALK"
    FAQ = "FAQ"
    CACHED = "CACHED"
    PERSONAL = "PERSONAL"
    RESEARCH = "RESEARCH"
    PRODUCT = "PRODUCT"
    RESTAURANT = "RESTAURANT"
    UNSAFE = "UNSAFE"


class RouteStage(StrEnum):
    """Which cascade stage made the call (arch.md 5.3)."""

    S0_RULES = "S0_RULES"
    S1_EXACT = "S1_EXACT"
    # S2 is hybrid retrieval: FULLTEXT plus, when enabled, vectors. The value
    # stays "S2_EMBEDDING" so historic trace rows keep their meaning, but the
    # name says RETRIEVAL — with embeddings off it is a pure lexical match, and
    # a stage labelled "embedding" in a trace sent people looking for a vector
    # search that never ran.
    S2_RETRIEVAL = "S2_EMBEDDING"
    S3_SEMANTIC_CACHE = "S3_SEMANTIC_CACHE"
    S4_LLM = "S4_LLM"
    FALLBACK = "FALLBACK"


class FaqCategory(StrEnum):
    GENERAL = "General"
    NUTRITION = "Nutrition"
    WORKOUT = "Workout"
    MEDICAL = "Medical"
    PRODUCT = "Product"
    APP_SUPPORT = "AppSupport"


class SafetyClass(StrEnum):
    INFORMATIONAL = "informational"
    GUIDANCE = "guidance"
    MEDICAL_SENSITIVE = "medical_sensitive"


class FaqStatus(StrEnum):
    DRAFT = "draft"
    REVIEW = "review"
    LIVE = "live"
    RETIRED = "retired"


class ConsentScope(StrEnum):
    VITALS = "vitals"
    NUTRITION = "nutrition"
    LABS = "labs"
    ACTIVITY = "activity"
    LOCATION = "location"
    PROFILE = "profile"


class CacheTier(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    MISS = "MISS"


class BlockType(StrEnum):
    """arch.md 10 — the machine-readable payload the app renders natively."""

    TEXT = "text"
    FAQ_ANSWER = "faq_answer"
    METRIC_CARD = "metric_card"
    INGREDIENT_TABLE = "ingredient_table"
    HAZARD_BADGE = "hazard_badge"
    EVIDENCE_LIST = "evidence_list"
    PRODUCT_UNIDENTIFIED = "product_unidentified"
    ACTION_PROMPT = "action_prompt"
    JOB_PENDING = "job_pending"
    SAFETY_NOTICE = "safety_notice"


class ProductUnidentifiedReason(StrEnum):
    """arch.md 8.7 keeps these three distinct because the user's next action differs."""

    UNREADABLE = "unreadable"
    NON_RETAIL_SYMBOL = "non_retail_symbol"
    UNLISTED_PRODUCT = "unlisted_product"


class Verdict(StrEnum):
    """arch.md 8.6 — the LLM may only select from this enum; rules produce it."""

    GENERALLY_SUITABLE = "Generally suitable"
    USE_WITH_CAUTION = "Use with caution"
    NOT_RECOMMENDED = "Not recommended for you"
    INSUFFICIENT_DATA = "Insufficient data"


class HazardLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionMethod(StrEnum):
    """How a raw label token became a chemical id (arch.md 8.3)."""

    EXACT_INCI = "exact_inci"
    CAS = "cas"
    EC = "ec"
    E_NUMBER = "e_number"
    SYNONYM = "synonym"
    FUZZY = "fuzzy"
    EMBEDDING = "embedding"
    UNRESOLVED = "unresolved"


class SourceTier(StrEnum):
    """arch.md 9.2."""

    T1_GOVERNMENT = "T1"
    T2_SYSTEMATIC = "T2"
    T3_PRIMARY = "T3"
    T4_SECONDARY = "T4"
    BLOCKED = "blocked"


class JobType(StrEnum):
    CHEMICAL_RESEARCH = "chemical_research"
    DEEP_RESEARCH = "deep_research"
    RESTAURANT_INVESTIGATION = "restaurant_investigation"
    MEMORY_SUMMARISE = "memory_summarise"
    PROFILE_CAPTURE = "profile_capture"
    CONTEXT_AGGREGATE = "context_aggregate"
    ETL_CHEMICAL_KB = "etl_chemical_kb"
    EMBEDDING_BACKFILL = "embedding_backfill"
    OCR = "ocr"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SafetyFlagKind(StrEnum):
    SELF_HARM = "self_harm"
    EMERGENCY_SYMPTOM = "emergency_symptom"
    MINOR = "minor"
    PROMPT_INJECTION = "prompt_injection"
    DISORDERED_EATING = "disordered_eating"
    PII_LEAK = "pii_leak"
    UNSUPPORTED_CLAIM = "unsupported_claim"
