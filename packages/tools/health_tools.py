"""Health data tools bound to one user (arch.md 6.2).

The prompt receives a *summary*; detail is reached through these. That is what
lets the payload stop carrying the whole dataset (arch.md P4) — the model asks
for what it needs instead of being handed everything.

The binding is structural: `HealthToolset(session, user_id)` constructs a
`HealthRepository` for that user, and every query inside it carries
`WHERE user_id = :uid`. There is no argument by which the model can name a
different user, because no tool takes one.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from packages.storage.repositories.conversation import ConversationRepository
from packages.storage.repositories.health import KNOWN_METRICS, HealthRepository

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_metrics_for_date",
            "description": "Every recorded health metric for one specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "ISO date, e.g. 2026-08-04"}
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_metric",
            "description": (
                "Mean, min, max and change vs the previous window for one metric. "
                "Use this rather than fetching every day and averaging yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": sorted(KNOWN_METRICS),
                        "description": "Which metric to summarise.",
                    },
                    "days": {"type": "integer", "description": "Window length, default 7."},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend",
            "description": "Daily values for one metric over recent days, oldest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": sorted(KNOWN_METRICS)},
                    "days": {"type": "integer", "description": "How many days, default 14."},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_activities",
            "description": "Recent workout sessions, optionally filtered by type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_type": {"type": "string", "description": "e.g. Running, CrossFit"},
                    "limit": {"type": "integer", "description": "Max sessions, default 10."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nutrition_day",
            "description": "Nutrition totals for one date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "ISO date."}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_labs",
            "description": (
                "The user's most recent medical report: BMI, blood pressure, HbA1c, "
                "lipids, conditions, allergies."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": "Compare one metric between two windows of equal length.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": sorted(KNOWN_METRICS)},
                    "days": {"type": "integer", "description": "Length of each window, default 7."},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "What the assistant has previously stored about this user: goals, restrictions, preferences.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value).strip()).date()
    except (TypeError, ValueError):
        return None


class HealthToolset:
    def __init__(self, session: Session, user_id: str) -> None:
        if not user_id:
            raise ValueError("HealthToolset requires a user_id")
        self.session = session
        self.user_id = user_id
        self.repo = HealthRepository(session, user_id)
        self.conversation = ConversationRepository(session)
        self.call_log: list[str] = []

    # -- dispatch ---------------------------------------------------------

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool. Never raises — an error becomes a result the model can
        read and recover from, rather than an exception that sinks the turn."""
        handler: Optional[Callable[..., dict[str, Any]]] = {
            "get_metrics_for_date": self.get_metrics_for_date,
            "summarise_metric": self.summarise_metric,
            "get_trend": self.get_trend,
            "list_activities": self.list_activities,
            "get_nutrition_day": self.get_nutrition_day,
            "get_latest_labs": self.get_latest_labs,
            "compare_periods": self.compare_periods,
            "recall_memory": self.recall_memory,
        }.get(name)

        if handler is None:
            return {"error": f"unknown tool '{name}'"}

        self.call_log.append(name)
        try:
            return handler(**args) if args else handler()
        except TypeError as exc:
            return {"error": f"bad arguments for '{name}': {exc}"}
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return {"error": f"tool '{name}' failed: {type(exc).__name__}"}

    # -- tools ------------------------------------------------------------

    def get_metrics_for_date(self, date: str) -> dict[str, Any]:
        parsed = _parse_date(date)
        if parsed is None:
            return {"error": f"could not parse date {date!r}; use YYYY-MM-DD"}
        points = self.repo.metrics_for_date(parsed)
        if not points:
            return {"date": str(parsed), "metrics": [], "note": "no data recorded for that date"}
        return {
            "date": str(parsed),
            "metrics": [
                {"metric": p.metric, "value": p.value, "unit": p.unit} for p in points
            ],
        }

    def summarise_metric(self, metric: str, days: int = 7) -> dict[str, Any]:
        if metric not in KNOWN_METRICS:
            return {"error": f"unknown metric {metric!r}", "known": sorted(KNOWN_METRICS)}
        stat = self.repo.metric_summary(metric, days=max(1, min(days, 365)))
        if stat is None:
            return {"metric": metric, "note": f"no {metric} data in the last {days} days"}
        return {
            "metric": metric,
            "window_days": stat.window_days,
            "mean": stat.mean,
            "min": stat.min,
            "max": stat.max,
            "change_vs_previous_window": stat.delta_vs_previous,
            "days_with_data": stat.sample_count,
        }

    def get_trend(self, metric: str, days: int = 14) -> dict[str, Any]:
        if metric not in KNOWN_METRICS:
            return {"error": f"unknown metric {metric!r}", "known": sorted(KNOWN_METRICS)}
        series = self.repo.metric_series(metric, days=max(1, min(days, 180)))
        return {
            "metric": metric,
            "days": days,
            "series": [
                {"date": str(p.measured_on), "value": p.value} for p in series if p.value is not None
            ],
        }

    def list_activities(self, activity_type: Optional[str] = None, limit: int = 10) -> dict[str, Any]:
        sessions = self.repo.recent_activities(activity_type, limit=max(1, min(limit, 50)))
        return {
            "activity_type": activity_type,
            "sessions": [
                {
                    "type": s.activity_type,
                    "started_at": str(s.started_at) if s.started_at else None,
                    "duration_min": s.duration_min,
                    "load": s.load,
                }
                for s in sessions
            ],
            "weekly_volume_by_type": self.repo.weekly_volume(7),
        }

    def get_nutrition_day(self, date: str) -> dict[str, Any]:
        parsed = _parse_date(date)
        if parsed is None:
            return {"error": f"could not parse date {date!r}; use YYYY-MM-DD"}
        points = self.repo.nutrition_day(parsed)
        if not points:
            return {"date": str(parsed), "note": "no nutrition logged for that date"}
        return {"date": str(parsed), "totals": {p.metric: p.value for p in points}}

    def get_latest_labs(self) -> dict[str, Any]:
        medical = self.repo.latest_medical()
        if medical.report_date is None:
            return {"note": "no medical report on file"}
        return {
            "report_date": str(medical.report_date),
            "bmi": medical.bmi,
            "blood_pressure": medical.blood_pressure,
            "hba1c": medical.hba1c,
            "lipids": medical.lipids,
            "conditions": medical.conditions,
            "allergies": medical.allergies,
            "flags": medical.flags,
        }

    def compare_periods(self, metric: str, days: int = 7) -> dict[str, Any]:
        if metric not in KNOWN_METRICS:
            return {"error": f"unknown metric {metric!r}", "known": sorted(KNOWN_METRICS)}
        window = max(1, min(days, 90))
        recent = self.repo.metric_summary(metric, days=window)
        if recent is None:
            return {"metric": metric, "note": "not enough data to compare"}
        return {
            "metric": metric,
            "window_days": window,
            "current_mean": recent.mean,
            "change_vs_previous_window": recent.delta_vs_previous,
            "direction": (
                "flat"
                if recent.delta_vs_previous in (None, 0)
                else ("up" if recent.delta_vs_previous > 0 else "down")
            ),
        }

    def recall_memory(self) -> dict[str, Any]:
        memories = self.conversation.recall(self.user_id)
        return {
            "memories": [{"kind": m["kind"], "value": m["value"]} for m in memories]
        }
