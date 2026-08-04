"""Seed a starter FAQ set.

arch.md 15 phase 3 seeds the KB from the top ~200 real questions; these are a
representative handful across categories so the router cascade, the template
renderer and the cache can be exercised end to end on a fresh database.

    python scripts/seed_faq.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.domain.enums import FaqCategory, FaqStatus, SafetyClass  # noqa: E402
from packages.domain.models import FaqItem, FaqVariants, PersonalisationRule  # noqa: E402
from packages.storage.db import session_scope  # noqa: E402
from packages.storage.repositories.faq import FaqRepository  # noqa: E402

NOW = datetime.now(timezone.utc)

SEED: list[FaqItem] = [
    FaqItem(
        id="faq_protein_target",
        status=FaqStatus.LIVE,
        category=FaqCategory.NUTRITION,
        canonical_question="How much protein should I eat per day?",
        paraphrases=[
            "what is my daily protein target",
            "how much protein do I need",
            "daily protein intake",
            "protein goal per day",
            "how many grams of protein should I have",
        ],
        answer_template=(
            "For general strength and body-composition goals, 1.6–2.2 g of protein per kg of "
            "bodyweight per day covers almost everyone. Spread it across 3–4 meals rather than "
            "loading it into one."
        ),
        variants=FaqVariants(
            with_data=(
                "At {{profile.weight_kg}} kg, aim for roughly 1.6–2.2 g of protein per kg — about "
                "{{profile.weight_kg}}×1.6 to {{profile.weight_kg}}×2.2 g a day. Spread it across "
                "3–4 meals rather than loading it into one."
            ),
            without_data=(
                "For general strength and body-composition goals, 1.6–2.2 g of protein per kg of "
                "bodyweight per day covers almost everyone. Tell me your weight and I'll give you "
                "the number directly."
            ),
        ),
        required_slots=["profile.weight_kg"],
        safety_class=SafetyClass.GUIDANCE,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
    FaqItem(
        id="faq_tdee",
        status=FaqStatus.LIVE,
        category=FaqCategory.NUTRITION,
        canonical_question="What is TDEE?",
        paraphrases=[
            "what does TDEE mean",
            "total daily energy expenditure",
            "explain TDEE",
            "TDEE vs BMR",
            "what is my BMR",
        ],
        answer_template=(
            "TDEE is Total Daily Energy Expenditure — everything you burn in a day. It is your BMR "
            "(what you'd burn at complete rest) plus the energy cost of digestion, daily movement, "
            "and training. It is the number a calorie target is set against: eat below it to lose "
            "weight, above it to gain."
        ),
        safety_class=SafetyClass.INFORMATIONAL,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
    FaqItem(
        id="faq_rest_days",
        status=FaqStatus.LIVE,
        category=FaqCategory.WORKOUT,
        canonical_question="How many rest days do I need per week?",
        paraphrases=[
            "how often should I rest",
            "do I need rest days",
            "how many days off from training",
            "rest day frequency",
        ],
        answer_template=(
            "Most people training 4–6 days a week do well with 1–2 full rest days, placed where "
            "your hardest sessions leave you most fatigued. Rest is when adaptation actually "
            "happens — it is part of the programme, not a gap in it."
        ),
        variants=FaqVariants(
            with_data=(
                "Most people training 4–6 days a week do well with 1–2 full rest days. Your recovery "
                "has been trending {{derived.trends.recovery}} recently, so plan them around your "
                "hardest sessions. Rest is when adaptation happens — it is part of the programme."
            )
        ),
        personalisation_rules=[
            PersonalisationRule(when="derived.deltas.recovery < 0", use_variant="with_data")
        ],
        safety_class=SafetyClass.GUIDANCE,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
    FaqItem(
        id="faq_sleep_recovery",
        status=FaqStatus.LIVE,
        category=FaqCategory.GENERAL,
        canonical_question="How does sleep affect recovery?",
        paraphrases=[
            "why is sleep important for recovery",
            "does sleep affect my recovery score",
            "sleep and muscle recovery",
            "how much sleep do I need to recover",
        ],
        answer_template=(
            "Sleep is the single largest lever on recovery. Most of your growth-hormone release and "
            "tissue repair happens during deep sleep, and short sleep raises resting heart rate, "
            "blunts strength, and makes perceived effort higher for the same work. 7–9 hours is the "
            "range where most people stop accumulating a deficit."
        ),
        safety_class=SafetyClass.INFORMATIONAL,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
    FaqItem(
        id="faq_scan_product",
        status=FaqStatus.LIVE,
        category=FaqCategory.PRODUCT,
        canonical_question="How do I scan a product?",
        paraphrases=[
            "how to scan a barcode",
            "how do I check a product",
            "scan an ingredient list",
            "how do I analyse a product label",
        ],
        answer_template=(
            "Point the camera at the barcode on the back of the pack. If the product isn't in any "
            "database, photograph the ingredient panel instead and I'll read the ingredients "
            "directly. Good light and a flat, filled frame make the biggest difference."
        ),
        safety_class=SafetyClass.INFORMATIONAL,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
    FaqItem(
        id="faq_data_privacy",
        status=FaqStatus.LIVE,
        category=FaqCategory.APP_SUPPORT,
        canonical_question="What data do you store about me?",
        paraphrases=[
            "what data do you keep",
            "is my health data private",
            "do you share my data",
            "how is my data used",
            "delete my data",
        ],
        answer_template=(
            "Your profile, synced health metrics, nutrition, activity and any medical reports you "
            "upload are stored on our servers and read only under the permissions you have granted. "
            "You can see exactly what's held with the 'context' action, review what I remember about "
            "you under memory, and revoke any permission at any time."
        ),
        safety_class=SafetyClass.INFORMATIONAL,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
    FaqItem(
        id="faq_bmi_meaning",
        status=FaqStatus.LIVE,
        category=FaqCategory.MEDICAL,
        canonical_question="What does my BMI mean?",
        paraphrases=[
            "explain BMI",
            "is my BMI healthy",
            "what is a good BMI",
            "BMI range",
        ],
        answer_template=(
            "BMI is weight divided by height squared. It is a population screening tool, not a "
            "diagnosis: it cannot tell muscle from fat, and the standard cut-offs fit South Asian "
            "populations poorly — risk starts to rise nearer 23 than 25. Treat it as one weak signal "
            "alongside waist measurement and blood markers."
        ),
        variants=FaqVariants(
            with_data=(
                "Your most recent BMI is {{medical.bmi}}. BMI is weight divided by height squared — a "
                "population screening tool, not a diagnosis: it cannot tell muscle from fat, and the "
                "standard cut-offs fit South Asian populations poorly, where risk starts to rise "
                "nearer 23 than 25. Treat it as one weak signal alongside waist measurement and "
                "blood markers, and discuss it with your doctor."
            )
        ),
        safety_class=SafetyClass.MEDICAL_SENSITIVE,
        owner="seed",
        reviewed_by="seed",
        reviewed_at=NOW,
    ),
]


def main() -> int:
    with session_scope() as session:
        repo = FaqRepository(session)
        total_surfaces = 0
        for item in SEED:
            repo.upsert_item(item)
            total_surfaces += repo.replace_surfaces(item.id, item)
            print(f"  seeded {item.id} ({item.category.value}, {len(item.paraphrases) + 1} surfaces)")

    print(f"\n{len(SEED)} FAQ items, {total_surfaces} surface forms.")
    print("Exact (S1) and fulltext (S2) routing work now. Embeddings are added")
    print("by the backfill once the chains package lands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
