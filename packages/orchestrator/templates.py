"""Deterministic response assembly (arch.md 5.4, 8.7, 10).

Two things here are load-bearing:

  * Personalisation is slot substitution and rule-selected variants, not a
    rewrite pass. `medical_sensitive` wording survives verbatim.
  * The product-not-found path is a typed block with a client action, not a
    system-prompt directive telling the model what to say (P7).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from packages.domain.enums import (
    BlockType,
    ProductUnidentifiedReason,
    SafetyClass,
    SafetyFlagKind,
    Verdict,
)
from packages.domain.models import (
    AnswerBlock,
    Citation,
    DataGap,
    Disclaimer,
    FaqItem,
    ProductAnalysis,
    UserContext,
)

logger = logging.getLogger(__name__)

_SLOT = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


# arch.md 10: the disclaimer policy is a rule table keyed on safety class, not
# something the model decides per turn.
DISCLAIMERS: dict[str, Disclaimer] = {
    "medical": Disclaimer(
        disclaimer_id="medical_general",
        text=(
            "This is general information, not medical advice. Talk to a qualified "
            "clinician about your own situation before acting on it."
        ),
        reason="safety_class=medical_sensitive",
    ),
    "guidance": Disclaimer(
        disclaimer_id="guidance_general",
        text="General guidance — adjust to how you actually feel and respond.",
        reason="safety_class=guidance",
    ),
    "product": Disclaimer(
        disclaimer_id="product_analysis",
        text=(
            "Ingredient assessments come from published hazard and regulatory data "
            "for each substance. They do not account for concentration, which the "
            "label rarely states."
        ),
        reason="content=product_analysis",
    ),
    "data_gap": Disclaimer(
        disclaimer_id="data_incomplete",
        text="Some of your data was unavailable, so this answer is partial.",
        reason="data_gaps present",
    ),
}


SAFETY_RESPONSES: dict[SafetyFlagKind, str] = {
    SafetyFlagKind.SELF_HARM: (
        "I'm not the right help for this, and I don't want to give you a canned answer "
        "when it matters. Please reach out to someone who can respond properly right now — "
        "in India, Tele-MANAS on 14416 or KIRAN on 1800-599-0019, both free and 24/7. "
        "If you're elsewhere, findahelpline.com lists local services. "
        "If you're in immediate danger, please call your local emergency number."
    ),
    SafetyFlagKind.EMERGENCY_SYMPTOM: (
        "What you're describing can be an emergency, and I'm not able to assess it. "
        "Please call your local emergency number now — 112 in India, 911 in the US, "
        "999 in the UK — or get to an emergency department. Don't wait to see if it passes."
    ),
    SafetyFlagKind.DISORDERED_EATING: (
        "I'm not going to help with that approach — it does real damage, and you deserve "
        "better support than I can give. If you're struggling with eating, a doctor or a "
        "specialist can actually help. In India, Tele-MANAS on 14416 is free and 24/7. "
        "I'm happy to talk about eating in a way that supports your training instead."
    ),
}


def _resolve_slot(path: str, context: Optional[UserContext]) -> Optional[Any]:
    """Walk a dotted path like `profile.weight_kg` against the context."""
    if context is None:
        return None
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def missing_slots(item: FaqItem, context: Optional[UserContext]) -> list[str]:
    return [slot for slot in item.required_slots if _resolve_slot(slot, context) is None]


def select_variant(item: FaqItem, context: Optional[UserContext]) -> str:
    """Rule-selected variant, falling back to the base template."""
    for rule in item.personalisation_rules:
        if rule.use_variant and _evaluate_condition(rule.when, context):
            candidate = getattr(item.variants, rule.use_variant, None)
            if candidate:
                return candidate

    if item.required_slots:
        has_all = not missing_slots(item, context)
        if has_all and item.variants.with_data:
            return item.variants.with_data
        if not has_all and item.variants.without_data:
            return item.variants.without_data

    return item.answer_template


def _evaluate_condition(expression: str, context: Optional[UserContext]) -> bool:
    """A deliberately tiny comparison language: `path op literal`.

    Not eval(). An authoring UI writes these, and an authoring UI that can run
    arbitrary Python in the request path is a vulnerability, not a feature.
    """
    if not expression or context is None:
        return False

    match = re.match(r"^\s*([a-zA-Z0-9_.]+)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$", expression)
    if not match:
        logger.warning("unparseable personalisation condition: %r", expression)
        return False

    path, op, raw_literal = match.groups()
    left = _resolve_slot(path, context)
    if left is None:
        return False

    literal: Any = raw_literal.strip().strip("'\"")
    try:
        right: Any = float(literal)
        left = float(left)
    except (TypeError, ValueError):
        right = literal
        left = str(left)

    try:
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
    except TypeError:
        return False
    return False


def render_faq(item: FaqItem, context: Optional[UserContext]) -> tuple[str, list[DataGap]]:
    """Substitute slots. Returns the text and any gaps that were left."""
    template = select_variant(item, context)
    gaps: list[DataGap] = []

    def substitute(match: re.Match[str]) -> str:
        path = match.group(1)
        value = _resolve_slot(path, context)
        if value is None:
            gaps.append(
                DataGap(
                    field=path,
                    why_it_matters="this answer is personalised on it",
                    how_to_supply="sync the relevant data or add it in your profile",
                )
            )
            return "—"
        return _format_value(value)

    return _SLOT.sub(substitute, template).strip(), gaps


def disclaimers_for(
    safety_class: SafetyClass,
    *,
    has_data_gaps: bool = False,
    is_product: bool = False,
) -> list[Disclaimer]:
    selected: list[Disclaimer] = []
    if safety_class is SafetyClass.MEDICAL_SENSITIVE:
        selected.append(DISCLAIMERS["medical"])
    elif safety_class is SafetyClass.GUIDANCE:
        selected.append(DISCLAIMERS["guidance"])
    if is_product:
        selected.append(DISCLAIMERS["product"])
    if has_data_gaps:
        selected.append(DISCLAIMERS["data_gap"])
    return selected


# -- deterministic blocks ---------------------------------------------------


def text_block(block_id: str, text: str) -> AnswerBlock:
    return AnswerBlock(block_id=block_id, type=BlockType.TEXT, text=text)


def faq_block(block_id: str, text: str, faq_id: str, version: int) -> AnswerBlock:
    return AnswerBlock(
        block_id=block_id,
        type=BlockType.FAQ_ANSWER,
        text=text,
        data={"faq_id": faq_id, "faq_version": version},
    )


def safety_block(kind: SafetyFlagKind) -> AnswerBlock:
    """Never a model call. The wording is reviewed and fixed."""
    return AnswerBlock(
        block_id="safety_1",
        type=BlockType.SAFETY_NOTICE,
        text=SAFETY_RESPONSES.get(
            kind,
            "I'm not able to help with this. Please speak to a qualified professional.",
        ),
        data={"kind": str(kind.value)},
    )


def product_unidentified_block(reason: ProductUnidentifiedReason) -> AnswerBlock:
    """arch.md 8.7. Three subtypes because the right user action differs; the
    client renders the action, so no prose directive is needed."""
    copy = {
        ProductUnidentifiedReason.UNREADABLE: (
            "I couldn't read that clearly enough.",
            "Retake the photo in better light, filling the frame with the label.",
            "retake_photo",
        ),
        ProductUnidentifiedReason.NON_RETAIL_SYMBOL: (
            "That code isn't a retail product barcode — it looks like an internal or "
            "logistics symbol.",
            "Photograph the ingredient panel on the back of the pack instead.",
            "photograph_ingredient_panel",
        ),
        ProductUnidentifiedReason.UNLISTED_PRODUCT: (
            "I read the barcode, but this product isn't in any database I can reach.",
            "Photograph the ingredient panel and I'll analyse it directly.",
            "photograph_ingredient_panel",
        ),
    }[reason]

    return AnswerBlock(
        block_id="product_unidentified_1",
        type=BlockType.PRODUCT_UNIDENTIFIED,
        text=f"{copy[0]} {copy[1]}",
        data={"reason": str(reason.value), "action": copy[2]},
    )


def job_pending_block(block_id: str, job_ids: list[str], what: str) -> AnswerBlock:
    return AnswerBlock(
        block_id=block_id,
        type=BlockType.JOB_PENDING,
        text=f"Still working on {what}. I'll have it shortly.",
        data={"job_ids": job_ids, "poll_action": "job_status"},
    )


# -- product analysis blocks (arch.md 8.6, 10) ------------------------------

_VERDICT_LEAD = {
    Verdict.GENERALLY_SUITABLE: "Nothing here conflicts with your profile.",
    Verdict.USE_WITH_CAUTION: "Worth knowing about before you use this.",
    Verdict.NOT_RECOMMENDED: "I'd avoid this one.",
    Verdict.INSUFFICIENT_DATA: "I couldn't read enough of the panel to judge this.",
}


def product_blocks(analysis: "ProductAnalysis") -> list[AnswerBlock]:
    """Structured blocks from the analysis. No model call anywhere in here.

    The verdict text comes from the enum the rules chose (arch.md 8.6); an LLM
    layer may later rewrite the prose in `text`, but `data.verdict` is what the
    client renders and what `guard_out` checks.
    """
    blocks: list[AnswerBlock] = []
    hazards_by_id = {h.chemical_id: h for h in analysis.hazards}
    flags_by_id: dict[str, list] = {}
    for flag in analysis.personal_flags:
        flags_by_id.setdefault(flag.chemical_id, []).append(flag)

    if analysis.identity and (analysis.identity.name or analysis.identity.brand):
        name = analysis.identity.name or "This product"
        brand = f" by {analysis.identity.brand}" if analysis.identity.brand else ""
        blocks.append(
            AnswerBlock(
                block_id="product_1",
                type=BlockType.TEXT,
                text=f"{name}{brand}.",
                data={
                    "barcode": analysis.identity.barcode,
                    "source": analysis.identity.source,
                    "confidence": analysis.identity.confidence,
                },
            )
        )

    blocks.append(
        AnswerBlock(
            block_id="verdict_1",
            type=BlockType.HAZARD_BADGE,
            text=_VERDICT_LEAD.get(analysis.verdict, ""),
            data={
                "verdict": str(analysis.verdict.value),
                "flag_count": len(analysis.personal_flags),
                "ingredient_count": len(analysis.ingredients),
                "unresolved_count": analysis.unresolved_count,
            },
        )
    )

    # Personal flags lead: they are the reason a specific person cares.
    if analysis.personal_flags:
        blocks.append(
            AnswerBlock(
                block_id="personal_1",
                type=BlockType.ACTION_PROMPT,
                text="Relevant to you specifically: "
                + "; ".join(
                    f"{f.display_name} — {f.reason}" for f in analysis.personal_flags[:5]
                )
                + ".",
                data={
                    "flags": [
                        {
                            "chemical_id": f.chemical_id,
                            "name": f.display_name,
                            "reason": f.reason,
                            "severity": str(f.severity.value),
                            "rule": f.source_of_rule,
                        }
                        for f in analysis.personal_flags
                    ]
                },
            )
        )

    rows = []
    for ingredient in analysis.ingredients:
        hazard = hazards_by_id.get(ingredient.chemical_id or "")
        rows.append(
            {
                "position": ingredient.position + 1,
                "name": ingredient.display_name or ingredient.raw_token,
                "raw": ingredient.raw_token,
                "recognised": ingredient.resolved,
                "resolution": str(ingredient.resolution_method.value),
                "confidence": ingredient.confidence,
                "hazard_level": str(hazard.hazard_level.value) if hazard else "unknown",
                "iarc_group": hazard.iarc_group if hazard else None,
                "endocrine": hazard.endocrine_flag if hazard else False,
                "allergen": hazard.allergen_flag if hazard else False,
                "banned_in": hazard.banned_in if hazard else [],
                "restricted_in": hazard.restricted_in if hazard else [],
                "caveat": hazard.concentration_caveat if hazard else None,
                "rules_fired": hazard.rule_ids if hazard else [],
                "personal_flags": [f.reason for f in flags_by_id.get(ingredient.chemical_id or "", [])],
            }
        )

    blocks.append(
        AnswerBlock(
            block_id="ingredients_1",
            type=BlockType.INGREDIENT_TABLE,
            text=f"{len(analysis.ingredients)} ingredients read from the panel.",
            data={"rows": rows, "kb_version": analysis.kb_version},
        )
    )

    # arch.md 8.3 step 4: unresolved tokens are surfaced, never silently dropped.
    if analysis.unresolved_count:
        unknown = [i.raw_token for i in analysis.ingredients if not i.resolved]
        blocks.append(
            AnswerBlock(
                block_id="unknown_1",
                type=BlockType.TEXT,
                text=(
                    f"{len(unknown)} ingredient(s) I don't recognise yet: "
                    + ", ".join(unknown[:8])
                    + ("…" if len(unknown) > 8 else "")
                    + ". They are not part of the assessment above."
                ),
                data={"tokens": unknown},
            )
        )

    if analysis.pending_job_ids:
        blocks.append(
            job_pending_block(
                "research_pending_1",
                analysis.pending_job_ids,
                f"{len(analysis.pending_job_ids)} unrecognised ingredient(s)",
            )
        )

    return blocks


def citations_for(analysis: "ProductAnalysis") -> list[Citation]:
    """Every hazard claim carries its source (arch.md 1.5)."""
    citations: list[Citation] = []
    seen: set[str] = set()

    for hazard in analysis.hazards:
        for ref in hazard.evidence:
            if ref.source_id in seen:
                continue
            seen.add(ref.source_id)
            citations.append(
                Citation(
                    citation_id=ref.source_id,
                    source=ref.title or ref.source_id,
                    tier=ref.tier,
                    url=ref.url,
                    title=ref.title,
                    retrieved_at=ref.retrieved_at,
                    supports_block_ids=["ingredients_1"],
                )
            )
    return citations
