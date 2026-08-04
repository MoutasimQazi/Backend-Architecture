"""Capture health, fitness and preference data the user states in conversation.

The app already has a `sync` action for data the *client* pushes — wearables,
manual entry, imports. This is the other half: what someone simply tells the
assistant. "I'm 78 kilos now", "I'm allergic to peanuts", "my knee still hurts
after the ACL repair" are all facts the system should hold, and none of them
arrive through `sync`.

Why this matters beyond convenience: the personalisation engine matches product
ingredients against `MedicalSnapshot.allergies`. An allergy mentioned only in
chat never reached that list, so a user could tell the assistant about a peanut
allergy on Monday and have it scan a peanut-containing product on Tuesday
without flagging it. Capturing it closes that gap.

Runs as a background job (arch.md 12) — never on the critical path, because the
answer should not wait on bookkeeping.

Three safety rules govern everything here:

  1. EXTRACTION IS ADDITIVE for medical facts. It may add an allergy; it may
     never remove one. A model that mis-parses "I'm not allergic to nuts any
     more" must not be able to delete a safety flag.
  2. Numbers are normalised in CODE, not by the model. The model reports the
     value and unit it saw; the conversion to canonical units is deterministic
     and testable.
  3. Nothing is written without consent for that scope.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from packages.chains.base import Chain
from packages.chains.providers import DataPolicy, ModelClass

logger = logging.getLogger(__name__)


# Canonical metric names. The long/narrow `health_metric` table takes anything,
# but the readers (trends, aggregates, the personal agent's tools) look for
# these exact names, so extraction must not invent new ones.
METRIC_NAMES = (
    "weight",
    "height",
    "body_fat",
    "resting_hr",
    "sleep",
    "steps",
    "waist",
    "vo2max",
    "systolic",
    "diastolic",
)

PREFERENCE_KINDS = (
    "goal",
    "dietary_restriction",
    "disliked_exercise",
    "preferred_exercise",
    "constraint",
    "injury",
    "schedule",
    "equipment",
    "motivation",
)


class CapturedMetric(BaseModel):
    metric: Literal[
        "weight", "height", "body_fat", "resting_hr",
        "sleep", "steps", "waist", "vo2max", "systolic", "diastolic",
    ]
    value: float
    unit: str = Field(default="", max_length=24)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class CapturedPreference(BaseModel):
    kind: Literal[
        "goal", "dietary_restriction", "disliked_exercise", "preferred_exercise",
        "constraint", "injury", "schedule", "equipment", "motivation",
    ]
    value: str = Field(max_length=200)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class CapturedMedical(BaseModel):
    allergies: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    pregnancy_status: Optional[
        Literal["pregnant", "breastfeeding", "trying", "none"]
    ] = None


class HealthCapture(BaseModel):
    metrics: list[CapturedMetric] = Field(default_factory=list)
    medical: CapturedMedical = Field(default_factory=CapturedMedical)
    preferences: list[CapturedPreference] = Field(default_factory=list)


CAPTURE_SYSTEM = """You extract durable facts a user has stated about their own
health, fitness and preferences, from their own messages.

Extract ONLY what the user asserted about themselves. Never extract:
- questions ("is creatine safe?" states nothing about them)
- anything the assistant said
- hypotheticals ("if I weighed 70kg...")
- goals stated as numbers they have NOT reached ("I want to be 70kg" is a goal,
  not a weight measurement)

METRICS — report the number and the unit exactly as the user gave them. Do not
convert. If they say "170 pounds", report value 170 unit "lb". If they say "5
foot 10", report value 70 unit "in". Only these metrics:
  weight, height, body_fat, resting_hr, sleep, steps, waist, vo2max,
  systolic, diastolic

MEDICAL — allergies, diagnosed conditions, medications they take, and pregnancy
status. Use the plain name only: "peanuts", not "allergic to peanuts". Include
an allergy even if mentioned in passing; it is safety-critical.

Only report a NEGATED medical fact ("I'm not diabetic") by omitting it. Never
list it.

PREFERENCES — durable, self-contained phrases that still make sense in six
months, with no pronouns. Kinds: goal, dietary_restriction, disliked_exercise,
preferred_exercise, constraint, injury, schedule, equipment, motivation.

Set confidence below 0.6 when the statement is vague or you are inferring.
Return empty lists when the user stated nothing durable. That is the common
case and it is a correct answer."""


# -- unit normalisation -----------------------------------------------------
#
# Deterministic, and deliberately not the model's job. A model asked to convert
# pounds to kilos will occasionally get it wrong, and a wrong weight silently
# corrupts every trend built on top of it.

_CANONICAL_UNIT = {
    "weight": "kg",
    "height": "cm",
    "waist": "cm",
    "body_fat": "%",
    "resting_hr": "bpm",
    "sleep": "h",
    "steps": "count",
    "vo2max": "ml/kg/min",
    "systolic": "mmHg",
    "diastolic": "mmHg",
}

_MASS_TO_KG = {"kg": 1.0, "kgs": 1.0, "kilo": 1.0, "kilos": 1.0, "kilogram": 1.0,
               "kilograms": 1.0, "lb": 0.45359237, "lbs": 0.45359237,
               "pound": 0.45359237, "pounds": 0.45359237,
               "st": 6.35029318, "stone": 6.35029318}

_LENGTH_TO_CM = {"cm": 1.0, "centimetre": 1.0, "centimeter": 1.0, "centimetres": 1.0,
                 "centimeters": 1.0, "m": 100.0, "metre": 100.0, "meter": 100.0,
                 "in": 2.54, "inch": 2.54, "inches": 2.54,
                 "ft": 30.48, "foot": 30.48, "feet": 30.48}

_TIME_TO_HOURS = {"h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
                  "m": 1 / 60, "min": 1 / 60, "mins": 1 / 60, "minute": 1 / 60,
                  "minutes": 1 / 60}

# Anything outside these is a misread, not a measurement. A "weight" of 7800
# is a typo or a step count that landed in the wrong field; writing it poisons
# the rolling averages that the assistant later reasons from.
_PLAUSIBLE = {
    "weight": (20.0, 400.0),
    "height": (50.0, 260.0),
    "waist": (30.0, 250.0),
    "body_fat": (2.0, 70.0),
    "resting_hr": (25.0, 140.0),
    "sleep": (0.0, 24.0),
    "steps": (0.0, 200_000.0),
    "vo2max": (10.0, 95.0),
    "systolic": (60.0, 260.0),
    "diastolic": (30.0, 180.0),
}


def normalise_metric(metric: str, value: float, unit: str) -> Optional[tuple[float, str]]:
    """Convert to the canonical unit. None means "do not store this".

    Returning None rather than storing the raw number is the point: an
    unconvertible or implausible reading is worse than a missing one, because
    everything downstream treats stored numbers as true.
    """
    unit_key = (unit or "").strip().lower().rstrip(".")

    if metric in ("weight",):
        factor = _MASS_TO_KG.get(unit_key)
        converted = value * factor if factor else (value if not unit_key else None)
    elif metric in ("height", "waist"):
        factor = _LENGTH_TO_CM.get(unit_key)
        # A bare height under 3 is metres ("I'm 1.75"), not centimetres.
        if factor is None and not unit_key:
            converted = value * 100.0 if value < 3.0 else value
        else:
            converted = value * factor if factor else None
    elif metric == "sleep":
        factor = _TIME_TO_HOURS.get(unit_key)
        converted = value * factor if factor else (value if not unit_key else None)
    elif metric == "body_fat":
        converted = value  # percent either way
    else:
        converted = value

    if converted is None:
        logger.info("dropping %s: unrecognised unit %r", metric, unit)
        return None

    low, high = _PLAUSIBLE.get(metric, (float("-inf"), float("inf")))
    if not (low <= converted <= high):
        logger.info(
            "dropping %s=%.2f%s: outside plausible range %s-%s",
            metric, converted, _CANONICAL_UNIT.get(metric, ""), low, high,
        )
        return None

    return round(converted, 4), _CANONICAL_UNIT.get(metric, unit_key or "")


# -- cheap pre-filter -------------------------------------------------------
#
# Enqueuing a capture job for "thanks!" would put an LLM call behind every
# greeting. This is the gate: a first-person statement with something concrete
# in it. False positives are fine (the job returns nothing); false negatives
# just mean the fact is caught on a later turn.

_SELF_REFERENCE = re.compile(
    r"\b(i|i'm|im|i am|i've|ive|i have|my|mine|me|myself)\b", re.IGNORECASE
)
_CONCRETE = re.compile(
    r"("
    r"\d"                                              # any number
    r"|\ballerg\w*|\bintoleran\w*|\bcoeliac\b|\bceliac\b"
    r"|\bdiabet\w*|\bhypertens\w*|\basthma\w*|\bthyroid\b|\bpcos\b"
    r"|\bpregnan\w*|\bbreastfeed\w*|\bnursing\b"
    r"|\bvegan\b|\bvegetarian\b|\bpescatarian\b|\bhalal\b|\bkosher\b|\bketo\b"
    r"|\bgluten\b|\blactose\b|\bdairy[- ]free\b|\bnut[- ]free\b"
    r"|\binjur\w*|\bsurgery\b|\bacl\b|\bsprain\w*|\bfractur\w*|\btendon\w*"
    r"|\bmedicat\w*|\btaking\b|\bprescrib\w*|\bstatin\w*|\bmetformin\b"
    r"|\bgoal\b|\btrying to\b|\bwant to\b|\baiming\b"
    r"|\bprefer\w*|\bdislike\w*|\bhate\b|\bavoid\w*|\bcan't eat\b|\bcannot eat\b"
    r"|\bgym\b|\bdumbbell\w*|\bbarbell\w*|\bkettlebell\w*|\btreadmill\b"
    r")",
    re.IGNORECASE,
)


def may_contain_profile_data(message: str) -> bool:
    """Whether a message is worth spending a capture job on."""
    if not message or len(message.strip()) < 8:
        return False
    return bool(_SELF_REFERENCE.search(message) and _CONCRETE.search(message))


class HealthCaptureChain(Chain):
    """Extracts metrics, medical facts and preferences from user messages."""

    name = "profile_capture"
    model_class = ModelClass.SMALL
    # Real health statements — allergies, conditions, medications. Identifiers
    # are masked by the base Chain; the clinical content is the payload.
    data_policy = DataPolicy.SENSITIVE
    temperature = 0.0
    max_tokens = 700
    system_prompt = CAPTURE_SYSTEM

    def capture(self, turns: list[dict[str, str]]) -> HealthCapture:
        user_turns = [t for t in turns if t.get("role") == "user"]
        if not user_turns:
            return HealthCapture()

        transcript = "\n".join(str(t.get("content") or "")[:600] for t in user_turns[-8:])
        result = self.run_structured(f"User messages:\n{transcript}", HealthCapture)

        if not result.ok or result.value is None:
            logger.info("profile capture unavailable (%s)", result.error)
            return HealthCapture()

        return result.value
