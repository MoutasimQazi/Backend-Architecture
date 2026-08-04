"""Restaurant analyzer (arch.md 11).

The two hard problems get the weight here, because they are what makes this
feature either useful or a liability:

**Entity disambiguation.** "Domino's" complaints in another city are not this
branch's. Every finding is tagged `branch` / `brand` / `unresolved`, and
unresolved findings are excluded from the score entirely — not down-weighted,
excluded. A finding we cannot bind to a place is not evidence about that place.

**Defamation exposure.** Output reports sourced facts with dates and links, and
never renders an unsourced verdict. Aggregate signals are stated as signals
("2 hygiene complaints in the last 12 months, both unverified user reports"),
never as fact about current conditions. "No adverse findings located" is a
first-class result — the absence of data is reported honestly rather than read
as a clean bill of health.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FindingScope(str, Enum):
    BRANCH = "branch"
    BRAND = "brand"
    UNRESOLVED = "unresolved"


class FindingKind(str, Enum):
    INSPECTION = "inspection"
    LICENCE = "licence"
    RECALL = "recall"
    NEWS = "news"
    COMPLAINT = "complaint"
    CLOSURE = "closure"


@dataclass
class Finding:
    kind: FindingKind
    scope: FindingScope
    summary: str
    source: str
    url: Optional[str] = None
    occurred_on: Optional[date] = None
    verified: bool = False
    severity: str = "info"

    @property
    def counts_towards_score(self) -> bool:
        """arch.md 11.2: unresolved findings are excluded from the score."""
        return self.scope is not FindingScope.UNRESOLVED

    @property
    def age_days(self) -> Optional[int]:
        if self.occurred_on is None:
            return None
        return (datetime.now(timezone.utc).date() - self.occurred_on).days


@dataclass
class ResolvedPlace:
    place_id: Optional[str] = None
    name: Optional[str] = None
    address: Optional[str] = None
    brand_id: Optional[str] = None
    confidence: float = 0.0

    @property
    def resolved(self) -> bool:
        return self.place_id is not None


@dataclass
class RestaurantReport:
    query: str
    place: ResolvedPlace = field(default_factory=ResolvedPlace)
    findings: list[Finding] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
    stages_unavailable: list[str] = field(default_factory=list)

    @property
    def scored_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.counts_towards_score]

    @property
    def excluded_findings(self) -> list[Finding]:
        return [f for f in self.findings if not f.counts_towards_score]

    @property
    def no_adverse_findings(self) -> bool:
        return not self.scored_findings

    def summary_line(self) -> str:
        """Signals stated as signals. Never a verdict about current conditions."""
        if not self.place.resolved:
            return (
                f"I couldn't identify a specific branch for '{self.query}', so I can't "
                "report anything reliable about it."
            )

        if self.no_adverse_findings:
            checked = ", ".join(self.stages_completed) or "the sources available"
            line = f"No adverse findings located for {self.place.name} in {checked}."
            if self.stages_unavailable:
                line += (
                    f" I could not check {', '.join(self.stages_unavailable)}, so this is not "
                    "a complete picture."
                )
            return line

        by_kind: dict[str, int] = {}
        for finding in self.scored_findings:
            by_kind[finding.kind.value] = by_kind.get(finding.kind.value, 0) + 1

        parts = []
        for kind, count in sorted(by_kind.items()):
            unverified = sum(
                1 for f in self.scored_findings if f.kind.value == kind and not f.verified
            )
            piece = f"{count} {kind} record{'s' if count > 1 else ''}"
            if unverified:
                piece += f" ({unverified} unverified)"
            parts.append(piece)

        return (
            f"For {self.place.name} I found {', '.join(parts)}. "
            "These are reports on record, not a judgement about conditions today."
        )

    def to_blocks(self) -> list[dict[str, Any]]:
        """Machine-readable payload. Every finding carries its source and date."""
        return [
            {
                "kind": f.kind.value,
                "scope": f.scope.value,
                "summary": f.summary,
                "source": f.source,
                "url": f.url,
                "date": str(f.occurred_on) if f.occurred_on else None,
                "verified": f.verified,
                "counts_towards_assessment": f.counts_towards_score,
            }
            for f in self.findings
        ]


class RestaurantAnalyzer:
    """Progressive: place → regulatory → recalls → news → complaints.

    Each stage records whether it ran. A stage that could not run is reported
    as unavailable rather than silently producing "nothing found", because
    those two states mean very different things to someone deciding where to
    eat.
    """

    def __init__(self, jurisdiction: str = "IN") -> None:
        self.jurisdiction = jurisdiction

    def analyze(self, query: str, place_id: Optional[str] = None) -> RestaurantReport:
        report = RestaurantReport(query=query)

        report.place = self._resolve_place(query, place_id)
        if not report.place.resolved:
            report.stages_unavailable.append("place resolution")
            return report

        report.stages_completed.append("place resolution")

        for stage_name, runner in (
            ("regulatory records", self._regulatory),
            ("recall notices", self._recalls),
            ("news", self._news),
            ("public complaints", self._complaints),
        ):
            findings, available = runner(report.place)
            if available:
                report.stages_completed.append(stage_name)
                report.findings.extend(findings)
            else:
                report.stages_unavailable.append(stage_name)

        return report

    # -- stages -----------------------------------------------------------

    def _resolve_place(self, query: str, place_id: Optional[str]) -> ResolvedPlace:
        """arch.md 11.1 step 1.

        Needs a Places provider. Without one, resolution fails — and a failed
        resolution must produce no findings at all, because every finding would
        be `unresolved` and excluded anyway. Reporting "nothing found" for an
        unidentified place would read as reassurance we have not earned.
        """
        if place_id:
            return ResolvedPlace(place_id=place_id, name=query, confidence=1.0)

        logger.info("place resolution unavailable: no Places provider configured")
        return ResolvedPlace()

    def _regulatory(self, place: ResolvedPlace) -> tuple[list[Finding], bool]:
        """Health-inspection open data; India: FoSCoS/FSSAI licence status.

        arch.md 16 Q6 flags that inspection open data varies wildly by city.
        Until a specific portal is wired, this reports unavailable rather than
        clean.
        """
        return [], False

    def _recalls(self, place: ResolvedPlace) -> tuple[list[Finding], bool]:
        return [], False

    def _news(self, place: ResolvedPlace) -> tuple[list[Finding], bool]:
        return [], False

    def _complaints(self, place: ResolvedPlace) -> tuple[list[Finding], bool]:
        """Review-corpus mining for illness/hygiene signals.

        arch.md 11.1 step 5 specifies a classifier rather than keyword match:
        "the chicken was sick" and "I was sick after the chicken" share
        keywords and mean entirely different things.
        """
        return [], False
