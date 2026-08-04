"""User-scoped health data access.

arch.md 6.2: the caller's `user_id` is bound at construction. Every statement in
this class carries `WHERE user_id = :uid`, so a tool built for one user cannot
address another user's rows even if the model asks it to. Tenancy is a
repository predicate, never a prompt instruction.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.domain.models import (
    Activity,
    ActivitySession,
    Derived,
    MedicalSnapshot,
    MetricPoint,
    Nutrition,
    RollingStat,
    SectionMeta,
    Vitals,
)

# Anything a caller can name has to be on this list. It stops a metric string
# reaching SQL and keeps error messages honest about what exists.
KNOWN_METRICS = {
    "recovery",
    "strain",
    "sleep",
    "sleep_hours",
    "rhr",
    "hrv",
    "weight",
    "steps",
    "spo2",
    "respiratory_rate",
    "body_fat",
    "vo2max",
}

NUTRITION_FIELDS = {
    "calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
    "water_ml",
    "diet_quality",
}


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _f(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


class HealthRepository:
    def __init__(self, session: Session, user_id: str) -> None:
        if not user_id:
            raise ValueError("HealthRepository requires a user_id")
        self.session = session
        self.user_id = user_id

    # -- raw reads --------------------------------------------------------

    def metrics_for_date(self, on: date) -> list[MetricPoint]:
        rows = self.session.execute(
            text(
                "SELECT metric, value, unit, measured_on FROM health_metric "
                "WHERE user_id = :uid AND measured_on = :d ORDER BY metric"
            ),
            {"uid": self.user_id, "d": on},
        ).mappings().all()
        return [
            MetricPoint(
                metric=r["metric"],
                value=_f(r["value"]),
                unit=r["unit"],
                measured_on=r["measured_on"],
            )
            for r in rows
        ]

    def latest_metrics(self) -> list[MetricPoint]:
        """One row per metric: the most recent measurement of each."""
        rows = self.session.execute(
            text(
                """
                SELECT hm.metric, hm.value, hm.unit, hm.measured_on
                FROM health_metric hm
                JOIN (
                    SELECT metric, MAX(measured_on) AS max_day
                    FROM health_metric WHERE user_id = :uid
                    GROUP BY metric
                ) latest ON latest.metric = hm.metric AND latest.max_day = hm.measured_on
                WHERE hm.user_id = :uid
                ORDER BY hm.metric
                """
            ),
            {"uid": self.user_id},
        ).mappings().all()
        return [
            MetricPoint(
                metric=r["metric"],
                value=_f(r["value"]),
                unit=r["unit"],
                measured_on=r["measured_on"],
            )
            for r in rows
        ]

    def metric_series(self, metric: str, days: int = 30) -> list[MetricPoint]:
        """Inclusive of today, so `days` dates are returned — see metric_summary."""
        if metric not in KNOWN_METRICS:
            return []
        since = date.today() - timedelta(days=days - 1)
        rows = self.session.execute(
            text(
                "SELECT metric, value, unit, measured_on FROM health_metric "
                "WHERE user_id = :uid AND metric = :m AND measured_on >= :since "
                "ORDER BY measured_on"
            ),
            {"uid": self.user_id, "m": metric, "since": since},
        ).mappings().all()
        return [
            MetricPoint(
                metric=r["metric"],
                value=_f(r["value"]),
                unit=r["unit"],
                measured_on=r["measured_on"],
            )
            for r in rows
        ]

    def metric_summary(self, metric: str, days: int = 7) -> Optional[RollingStat]:
        """Aggregated in SQL — the rows never come back over the wire.

        The window is inclusive of today, so `days=7` spans exactly 7 dates.
        `today - 7` would span 8 and quietly inflate every rolling average by
        one extra day.
        """
        if metric not in KNOWN_METRICS:
            return None
        since = date.today() - timedelta(days=days - 1)
        row = self.session.execute(
            text(
                "SELECT AVG(value) AS mean, MIN(value) AS lo, MAX(value) AS hi, COUNT(*) AS n "
                "FROM health_metric "
                "WHERE user_id = :uid AND metric = :m AND measured_on >= :since AND value IS NOT NULL"
            ),
            {"uid": self.user_id, "m": metric, "since": since},
        ).mappings().first()

        if not row or not row["n"]:
            return None

        prev = self.session.execute(
            text(
                "SELECT AVG(value) AS mean FROM health_metric "
                "WHERE user_id = :uid AND metric = :m "
                "  AND measured_on >= :prev_since AND measured_on < :since AND value IS NOT NULL"
            ),
            {
                "uid": self.user_id,
                "m": metric,
                "prev_since": since - timedelta(days=days),
                "since": since,
            },
        ).mappings().first()

        mean = _f(row["mean"])
        prev_mean = _f(prev["mean"]) if prev else None
        delta = None
        if mean is not None and prev_mean is not None:
            delta = round(mean - prev_mean, 4)

        return RollingStat(
            metric=metric,
            window_days=days,
            mean=round(mean, 4) if mean is not None else None,
            min=_f(row["lo"]),
            max=_f(row["hi"]),
            delta_vs_previous=delta,
            sample_count=int(row["n"]),
        )

    def nutrition_day(self, on: date) -> list[MetricPoint]:
        row = self.session.execute(
            text(
                "SELECT calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, "
                "       water_ml, diet_quality, consumed_on "
                "FROM nutrition_day WHERE user_id = :uid AND consumed_on = :d"
            ),
            {"uid": self.user_id, "d": on},
        ).mappings().first()
        if not row:
            return []
        return [
            MetricPoint(metric=field, value=_f(row[field]), measured_on=row["consumed_on"])
            for field in NUTRITION_FIELDS
            if row[field] is not None
        ]

    def nutrition_rolling(self, days: int = 7) -> list[RollingStat]:
        """Inclusive of today, so `days` dates are averaged — see metric_summary."""
        since = date.today() - timedelta(days=days - 1)
        row = self.session.execute(
            text(
                "SELECT AVG(calories) c, AVG(protein_g) p, AVG(carbs_g) cb, AVG(fat_g) f, "
                "       AVG(fiber_g) fb, AVG(sugar_g) s, AVG(sodium_mg) so, COUNT(*) n "
                "FROM nutrition_day WHERE user_id = :uid AND consumed_on >= :since"
            ),
            {"uid": self.user_id, "since": since},
        ).mappings().first()

        if not row or not row["n"]:
            return []

        mapping = {
            "calories": row["c"],
            "protein_g": row["p"],
            "carbs_g": row["cb"],
            "fat_g": row["f"],
            "fiber_g": row["fb"],
            "sugar_g": row["s"],
            "sodium_mg": row["so"],
        }
        return [
            RollingStat(
                metric=name,
                window_days=days,
                mean=round(float(value), 2),
                sample_count=int(row["n"]),
            )
            for name, value in mapping.items()
            if value is not None
        ]

    def recent_activities(self, activity_type: Optional[str] = None, limit: int = 10) -> list[ActivitySession]:
        limit = max(1, min(limit, 100))
        clause = "AND activity_type = :atype" if activity_type else ""
        params: dict[str, Any] = {"uid": self.user_id, "lim": limit}
        if activity_type:
            params["atype"] = activity_type

        rows = self.session.execute(
            text(
                f"SELECT activity_type, started_at, duration_min, load_score "
                f"FROM activity_session WHERE user_id = :uid {clause} "
                f"ORDER BY started_at DESC LIMIT :lim"
            ),
            params,
        ).mappings().all()
        return [
            ActivitySession(
                activity_type=r["activity_type"],
                started_at=r["started_at"],
                duration_min=_f(r["duration_min"]),
                load=_f(r["load_score"]),
            )
            for r in rows
        ]

    def weekly_volume(self, days: int = 7) -> dict[str, float]:
        rows = self.session.execute(
            text(
                "SELECT activity_type, SUM(duration_min) AS total FROM activity_session "
                "WHERE user_id = :uid AND started_at >= (NOW() - INTERVAL :d DAY) "
                "GROUP BY activity_type"
            ),
            {"uid": self.user_id, "d": days},
        ).mappings().all()
        return {r["activity_type"]: float(r["total"] or 0) for r in rows}

    def latest_medical(self) -> MedicalSnapshot:
        row = self.session.execute(
            text(
                "SELECT report_date, bmi, systolic, diastolic, hba1c, lipids, flags, "
                "       conditions, allergies, medications "
                "FROM medical_report WHERE user_id = :uid ORDER BY report_date DESC LIMIT 1"
            ),
            {"uid": self.user_id},
        ).mappings().first()

        if not row:
            return MedicalSnapshot()

        bp = None
        if row["systolic"] is not None and row["diastolic"] is not None:
            bp = f"{int(row['systolic'])}/{int(row['diastolic'])}"

        return MedicalSnapshot(
            report_date=row["report_date"],
            bmi=_f(row["bmi"]),
            blood_pressure=bp,
            hba1c=_f(row["hba1c"]),
            lipids=_loads(row["lipids"]) or {},
            flags=_loads(row["flags"]) or [],
            conditions=_loads(row["conditions"]) or [],
            allergies=_loads(row["allergies"]) or [],
            medications=_loads(row["medications"]) or [],
        )

    # -- precomputed aggregates (arch.md 6.1) -----------------------------

    def get_aggregate(self, section: str) -> Optional[tuple[dict[str, Any], SectionMeta]]:
        row = self.session.execute(
            text(
                "SELECT payload, completeness, fresh_as_of, version FROM user_aggregate "
                "WHERE user_id = :uid AND section = :section"
            ),
            {"uid": self.user_id, "section": section},
        ).mappings().first()
        if not row:
            return None
        payload = _loads(row["payload"]) or {}
        meta = SectionMeta(
            fresh_as_of=row["fresh_as_of"],
            completeness=float(row["completeness"] or 0),
            version=str(row["version"]),
        )
        return payload, meta

    def put_aggregate(
        self,
        section: str,
        payload: dict[str, Any],
        completeness: float = 0.0,
        fresh_as_of: Optional[Any] = None,
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO user_aggregate (user_id, section, payload, completeness, fresh_as_of, version)
                VALUES (:uid, :section, :payload, :completeness, :fresh, 1)
                ON DUPLICATE KEY UPDATE
                    payload = VALUES(payload),
                    completeness = VALUES(completeness),
                    fresh_as_of = VALUES(fresh_as_of),
                    version = version + 1
                """
            ),
            {
                "uid": self.user_id,
                "section": section,
                "payload": json.dumps(payload, default=str),
                "completeness": completeness,
                "fresh": fresh_as_of,
            },
        )

    def aggregate_versions(self) -> dict[str, str]:
        """Section -> version. Feeds the cache `context_fingerprint` so a
        nutrition answer is not invalidated by a workout sync (arch.md 7.2)."""
        rows = self.session.execute(
            text("SELECT section, version FROM user_aggregate WHERE user_id = :uid"),
            {"uid": self.user_id},
        ).all()
        return {section: str(version) for section, version in rows}

    # -- section builders used by context_build ---------------------------

    def build_vitals(self) -> Vitals:
        latest = [m for m in self.latest_metrics() if m.metric in KNOWN_METRICS]
        rolling: list[RollingStat] = []
        for metric in ("recovery", "strain", "sleep_hours", "rhr", "weight"):
            for window in (7, 30):
                stat = self.metric_summary(metric, days=window)
                if stat:
                    rolling.append(stat)
        return Vitals(latest=latest, rolling=rolling)

    def build_nutrition(self) -> Nutrition:
        return Nutrition(
            latest_day=self.nutrition_day(date.today()) or self.nutrition_day(date.today() - timedelta(days=1)),
            rolling=self.nutrition_rolling(7),
        )

    def build_activity(self) -> Activity:
        return Activity(
            recent=self.recent_activities(limit=10),
            weekly_volume_by_type=self.weekly_volume(7),
        )

    def build_derived(self) -> Derived:
        trends: dict[str, str] = {}
        deltas: dict[str, float] = {}
        for metric in ("weight", "recovery", "sleep_hours", "rhr"):
            stat = self.metric_summary(metric, days=7)
            if not stat or stat.delta_vs_previous is None:
                continue
            deltas[metric] = stat.delta_vs_previous
            if abs(stat.delta_vs_previous) < 1e-6:
                trends[metric] = "flat"
            else:
                trends[metric] = "up" if stat.delta_vs_previous > 0 else "down"
        return Derived(trends=trends, deltas=deltas)
