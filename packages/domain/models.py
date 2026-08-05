"""The shared contract.

Every node in the graph reads and writes these types; nothing passes untyped
dicts between services. arch.md 4.1 (state), 6.1 (user context), 8 (product),
9 (evidence), 10 (response assembly).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from packages.domain.enums import (
    BlockType,
    CacheTier,
    ConsentScope,
    FaqCategory,
    FaqStatus,
    HazardLevel,
    JobStatus,
    JobType,
    ProductUnidentifiedReason,
    ResolutionMethod,
    RouteLabel,
    RouteStage,
    SafetyClass,
    SafetyFlagKind,
    Severity,
    SourceTier,
    Verdict,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=True)


# --------------------------------------------------------------------------
# Request envelope (arch.md 3: the payload no longer carries the dataset)
# --------------------------------------------------------------------------


class Attachment(Base):
    """A handle, never bytes. `ingest` stores the blob and hands on the id."""

    attachment_id: str
    mime_type: str
    size_bytes: int = 0
    sha256: Optional[str] = None
    filename: Optional[str] = None
    page_index: Optional[int] = None  # set when a PDF page was rasterised
    source_attachment_id: Optional[str] = None


class RequestInfo(Base):
    user_id: str
    session_id: str
    turn_id: str
    locale: str = "en"
    client_version: Optional[str] = None
    jurisdiction: str = "IN"  # drives which regulatory tables apply (arch.md 8.5)
    received_at: datetime = Field(default_factory=utcnow)


class InputData(Base):
    text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    client_hints: dict[str, Any] = Field(default_factory=dict)


class ConsentState(Base):
    granted_scopes: list[ConsentScope] = Field(default_factory=list)
    masking_policy: Literal["strict", "standard", "off"] = "standard"

    def allows(self, scope: ConsentScope) -> bool:
        return scope in self.granted_scopes


# --------------------------------------------------------------------------
# User context (arch.md 6.1) — a summary, with detail reached through tools
# --------------------------------------------------------------------------


class SectionMeta(Base):
    fresh_as_of: Optional[datetime] = None
    completeness: float = 0.0  # 0..1
    withheld: bool = False
    withheld_reason: Optional[str] = None
    version: Optional[str] = None  # feeds the cache context_fingerprint


class Profile(Base):
    display_name: Optional[str] = None
    age_band: Optional[str] = None
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    goals: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    pregnancy_status: Optional[str] = None


class MetricPoint(Base):
    metric: str
    value: Optional[float] = None
    unit: Optional[str] = None
    measured_on: Optional[date] = None


class RollingStat(Base):
    metric: str
    window_days: int
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    delta_vs_previous: Optional[float] = None
    sample_count: int = 0


class Vitals(Base):
    latest: list[MetricPoint] = Field(default_factory=list)
    rolling: list[RollingStat] = Field(default_factory=list)


class Nutrition(Base):
    latest_day: list[MetricPoint] = Field(default_factory=list)
    rolling: list[RollingStat] = Field(default_factory=list)
    diet_quality_trend: Optional[str] = None


class MedicalSnapshot(Base):
    report_date: Optional[date] = None
    bmi: Optional[float] = None
    blood_pressure: Optional[str] = None
    hba1c: Optional[float] = None
    lipids: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


class ActivitySession(Base):
    activity_type: str
    started_at: Optional[datetime] = None
    duration_min: Optional[float] = None
    load: Optional[float] = None


class Activity(Base):
    recent: list[ActivitySession] = Field(default_factory=list)
    weekly_volume_by_type: dict[str, float] = Field(default_factory=dict)


class Derived(Base):
    trends: dict[str, str] = Field(default_factory=dict)
    streaks: dict[str, int] = Field(default_factory=dict)
    deltas: dict[str, float] = Field(default_factory=dict)
    adherence: Optional[float] = None
    anomalies: list[str] = Field(default_factory=list)


class UserContext(Base):
    user_id: str
    profile: Profile = Field(default_factory=Profile)
    vitals: Vitals = Field(default_factory=Vitals)
    nutrition: Nutrition = Field(default_factory=Nutrition)
    medical: MedicalSnapshot = Field(default_factory=MedicalSnapshot)
    activity: Activity = Field(default_factory=Activity)
    derived: Derived = Field(default_factory=Derived)
    meta: dict[str, SectionMeta] = Field(default_factory=dict)

    def sections_read(self) -> list[str]:
        """Sections that actually contributed — the basis of the cache
        fingerprint, so a workout sync does not bust a nutrition answer."""
        return sorted(name for name, meta in self.meta.items() if not meta.withheld)

    def fingerprint_parts(self) -> dict[str, str]:
        return {
            name: (meta.version or "0")
            for name, meta in sorted(self.meta.items())
            if not meta.withheld
        }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


class RouteDecision(Base):
    label: RouteLabel
    confidence: float = 0.0
    stage: RouteStage = RouteStage.S0_RULES
    rationale: str = ""
    fallbacks: list[RouteLabel] = Field(default_factory=list)
    category: Optional[FaqCategory] = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# --------------------------------------------------------------------------
# FAQ (arch.md 5.1)
# --------------------------------------------------------------------------


class FaqVariants(Base):
    with_data: Optional[str] = None
    without_data: Optional[str] = None
    short: Optional[str] = None
    long: Optional[str] = None


class PersonalisationRule(Base):
    when: str  # a small expression evaluated against UserContext, e.g. "medical.bmi >= 30"
    use_variant: Optional[str] = None
    append_block: Optional[str] = None


class FaqItem(Base):
    id: str
    version: int = 1
    status: FaqStatus = FaqStatus.DRAFT
    category: FaqCategory = FaqCategory.GENERAL
    canonical_question: str
    paraphrases: list[str] = Field(default_factory=list)
    answer_template: str = ""
    variants: FaqVariants = Field(default_factory=FaqVariants)
    required_slots: list[str] = Field(default_factory=list)
    personalisation_rules: list[PersonalisationRule] = Field(default_factory=list)
    safety_class: SafetyClass = SafetyClass.INFORMATIONAL
    locale: str = "en"
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    owner: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class FaqHit(Base):
    faq_id: str
    score: float
    matched_surface: str
    stage: RouteStage
    item: Optional[FaqItem] = None


# --------------------------------------------------------------------------
# Product analysis (arch.md 8)
# --------------------------------------------------------------------------


class ResolvedIngredient(Base):
    raw_token: str
    position: int
    chemical_id: Optional[str] = None
    display_name: Optional[str] = None
    confidence: float = 0.0
    resolution_method: ResolutionMethod = ResolutionMethod.UNRESOLVED
    qualifiers: list[str] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.chemical_id is not None


class EvidenceRef(Base):
    source_id: str
    tier: SourceTier
    title: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    study_design: Optional[str] = None
    independence: Optional[float] = None  # arch.md 9.3 — shown, not used to filter
    retrieved_at: Optional[datetime] = None


class HazardFinding(Base):
    """Produced by the deterministic rules engine. The LLM never writes these."""

    chemical_id: str
    display_name: str
    hazard_level: HazardLevel = HazardLevel.UNKNOWN
    ghs_codes: list[str] = Field(default_factory=list)
    iarc_group: Optional[str] = None
    endocrine_flag: bool = False
    endocrine_lists: list[str] = Field(default_factory=list)
    allergen_flag: bool = False
    restricted_in: list[str] = Field(default_factory=list)
    banned_in: list[str] = Field(default_factory=list)
    concentration_caveat: Optional[str] = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    rules_version: str = "v1"


class PersonalFlag(Base):
    chemical_id: str
    display_name: str
    reason: str
    severity: Severity = Severity.INFO
    source_of_rule: str = ""
    # Which ingredient on the panel raised this, by its position.
    #
    # chemical_id is not sufficient to identify it: an unresolved ingredient
    # has no chemical_id, so grouping flags by that key put every unresolved
    # ingredient in one bucket. A single groundnut warning was then rendered
    # against all twelve unresolved rows including wheat flour and ferric
    # pyrophosphate — an alert on everything, which teaches people to ignore
    # the one that matters.
    position: Optional[int] = None


class ProductIdentity(Base):
    product_id: Optional[str] = None
    barcode: Optional[str] = None
    barcode_format: Optional[str] = None
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    confidence: float = 0.0
    fetched_at: Optional[datetime] = None


class ProductAnalysis(Base):
    identity: Optional[ProductIdentity] = None
    unidentified_reason: Optional[ProductUnidentifiedReason] = None
    ingredients: list[ResolvedIngredient] = Field(default_factory=list)
    hazards: list[HazardFinding] = Field(default_factory=list)
    personal_flags: list[PersonalFlag] = Field(default_factory=list)
    verdict: Verdict = Verdict.INSUFFICIENT_DATA
    pending_chemical_ids: list[str] = Field(default_factory=list)
    pending_job_ids: list[str] = Field(default_factory=list)
    kb_version: str = "v1"

    @property
    def unresolved_count(self) -> int:
        return sum(1 for i in self.ingredients if not i.resolved)


# --------------------------------------------------------------------------
# Response assembly (arch.md 10)
# --------------------------------------------------------------------------


class AnswerBlock(Base):
    block_id: str
    type: BlockType
    text: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class Citation(Base):
    citation_id: str
    source: str
    tier: SourceTier = SourceTier.T4_SECONDARY
    url: Optional[str] = None
    title: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    supports_block_ids: list[str] = Field(default_factory=list)


class Disclaimer(Base):
    disclaimer_id: str
    text: str
    reason: str = ""


class DataGap(Base):
    field: str
    why_it_matters: str = ""
    how_to_supply: str = ""


class AnswerPayload(Base):
    blocks: list[AnswerBlock] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    disclaimers: list[Disclaimer] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_reason: Optional[str] = None
    data_gaps: list[DataGap] = Field(default_factory=list)
    route_debug: Optional[dict[str, Any]] = None  # stripped before it leaves the API

    def rendered_text(self) -> str:
        parts = [b.text for b in self.blocks if b.text]
        return "\n\n".join(parts).strip()


# --------------------------------------------------------------------------
# Telemetry, safety, and the state object itself (arch.md 4.1)
# --------------------------------------------------------------------------


class NodeTiming(Base):
    node: str
    started_at: datetime
    duration_ms: float
    ok: bool = True
    error: Optional[str] = None


class TokenCost(Base):
    node: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0


class Telemetry(Base):
    node_timings: list[NodeTiming] = Field(default_factory=list)
    token_costs: list[TokenCost] = Field(default_factory=list)
    cache_status: CacheTier = CacheTier.MISS
    cache_key: Optional[str] = None

    @property
    def total_usd(self) -> float:
        return round(sum(c.usd for c in self.token_costs), 6)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.token_costs)


class SafetyFlag(Base):
    kind: SafetyFlagKind
    detail: str = ""
    severity: Severity = Severity.INFO
    blocking: bool = False


class Candidates(Base):
    faq_hits: list[FaqHit] = Field(default_factory=list)
    cache_hits: list[dict[str, Any]] = Field(default_factory=list)
    evidence_docs: list[EvidenceRef] = Field(default_factory=list)
    analyzer_result: Optional[ProductAnalysis] = None


class Draft(Base):
    answer_blocks: list[AnswerBlock] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    disclaimers: list[Disclaimer] = Field(default_factory=list)


class Flags(Base):
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    degraded_sources: list[str] = Field(default_factory=list)


class ConversationState(Base):
    """Carried between LangGraph nodes. A node writes only the fields it owns."""

    request: RequestInfo
    input: InputData = Field(default_factory=InputData)
    consent: ConsentState = Field(default_factory=ConsentState)
    context: Optional[UserContext] = None
    route: Optional[RouteDecision] = None
    candidates: Candidates = Field(default_factory=Candidates)
    draft: Draft = Field(default_factory=Draft)
    payload: Optional[AnswerPayload] = None
    telemetry: Telemetry = Field(default_factory=Telemetry)
    flags: Flags = Field(default_factory=Flags)

    @property
    def blocked(self) -> bool:
        return any(f.blocking for f in self.flags.safety_flags)


# --------------------------------------------------------------------------
# Jobs (arch.md 2 async plane — DB-backed here, no broker)
# --------------------------------------------------------------------------


class JobRecord(Base):
    job_id: str
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    user_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 100
    available_at: datetime = Field(default_factory=utcnow)
    lease_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Public API DTOs
# --------------------------------------------------------------------------


class ChatRequest(Base):
    """arch.md 3: only ids and the message. No csv_health_data."""

    message: str = ""
    session_id: Optional[str] = None
    attachments: list[Attachment] = Field(default_factory=list)
    locale: str = "en"
    client_version: Optional[str] = None
    client_hints: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class ChatResponse(Base):
    success: bool = True
    turn_id: str
    session_id: str
    message: str
    payload: AnswerPayload
    route: Optional[RouteDecision] = None
    cache: CacheTier = CacheTier.MISS
    pending_jobs: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
