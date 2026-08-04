"""Seed a starter Chemical KB and the hazard rule set.

arch.md 15 phase 4 targets the top ~5,000 INCI/food ingredients via ETL. This
is the shape of that data plus enough real coverage to exercise every branch of
the pipeline: a resolved-and-clean ingredient, a restricted one, a listed
carcinogen, an endocrine-disruptor, an EU-26 fragrance allergen, and an
E-number.

Every assertion carries its source. Nothing here is invented — but it is a
starter set, not a reviewed dossier collection, so `review_status` stays
'published' only because the runtime needs it; a real deployment reviews these.

    python scripts/seed_chemicals.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.product.rules import DEFAULT_RULES  # noqa: E402
from packages.storage.db import session_scope  # noqa: E402
from packages.storage.repositories.chemicals import ChemicalRepository  # noqa: E402

# (chemical_id, display, inci, cas, e_number, class, functions, synonyms, assertions)
CHEMICALS: list[dict] = [
    {
        "chemical_id": "aqua",
        "display_name": "Water",
        "inci_name": "AQUA",
        "cas": "7732-18-5",
        "chem_class": "solvent",
        "functions": ["solvent"],
        "synonyms": ["water", "eau", "purified water", "aqua/water/eau"],
        "assertions": [],
    },
    {
        "chemical_id": "glycerin",
        "display_name": "Glycerin",
        "inci_name": "GLYCERIN",
        "cas": "56-81-5",
        "e_number": "E422",
        "chem_class": "polyol",
        "functions": ["humectant", "solvent"],
        "synonyms": ["glycerine", "glycerol", "e422", "1,2,3-propanetriol"],
        "assertions": [],
    },
    {
        "chemical_id": "sodium_lauryl_sulfate",
        "display_name": "Sodium Lauryl Sulfate",
        "inci_name": "SODIUM LAURYL SULFATE",
        "cas": "151-21-3",
        "chem_class": "anionic surfactant",
        "functions": ["surfactant", "cleansing"],
        "synonyms": ["sls", "sodium dodecyl sulfate", "sodium laurilsulfate"],
        "assertions": [
            {
                "domain": "hazard",
                "key_name": "ghs_code",
                "value": "H315",
                "source": "ECHA CLP",
                "evidence_grade": "A",
            },
            {
                "domain": "hazard",
                "key_name": "ghs_code",
                "value": "H319",
                "source": "ECHA CLP",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "phenoxyethanol",
        "display_name": "Phenoxyethanol",
        "inci_name": "PHENOXYETHANOL",
        "cas": "122-99-6",
        "chem_class": "glycol ether",
        "functions": ["preservative"],
        "synonyms": ["2-phenoxyethanol", "ethylene glycol monophenyl ether"],
        "assertions": [
            {
                "domain": "regulatory",
                "key_name": "annex_iii",
                "value": "max 1.0%",
                "jurisdiction": "EU",
                "product_class": "cosmetic",
                "limit_value": 1.0,
                "limit_unit": "%",
                "source": "EU Cosmetics Regulation Annex III",
                "evidence_grade": "A",
            },
            {
                "domain": "hazard",
                "key_name": "ghs_code",
                "value": "H319",
                "source": "ECHA CLP",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "formaldehyde",
        "display_name": "Formaldehyde",
        "inci_name": "FORMALDEHYDE",
        "cas": "50-00-0",
        "chem_class": "aldehyde",
        "functions": ["preservative"],
        "synonyms": ["formalin", "methanal", "formic aldehyde"],
        "assertions": [
            {
                "domain": "hazard",
                "key_name": "iarc_group",
                "value": "1",
                "source": "IARC Monographs Vol. 100F",
                "evidence_grade": "A",
            },
            {
                "domain": "hazard",
                "key_name": "ghs_code",
                "value": "H350",
                "source": "ECHA CLP",
                "evidence_grade": "A",
            },
            {
                "domain": "regulatory",
                "key_name": "annex_ii",
                "value": "prohibited in cosmetics",
                "jurisdiction": "EU",
                "product_class": "cosmetic",
                "source": "EU Cosmetics Regulation Annex II",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "butylparaben",
        "display_name": "Butylparaben",
        "inci_name": "BUTYLPARABEN",
        "cas": "94-26-8",
        "chem_class": "paraben",
        "functions": ["preservative"],
        "synonyms": ["butyl paraben", "butyl 4-hydroxybenzoate", "butyl parahydroxybenzoate"],
        "assertions": [
            {
                "domain": "endocrine",
                "key_name": "list_membership",
                "value": "EU EDC candidate list",
                "jurisdiction": "EU",
                "source": "EU Endocrine Disruptor priority list",
                "evidence_grade": "B",
            },
            {
                "domain": "regulatory",
                "key_name": "annex_iii",
                "value": "max 0.14% as acid",
                "jurisdiction": "EU",
                "product_class": "cosmetic",
                "limit_value": 0.14,
                "limit_unit": "%",
                "source": "EU Cosmetics Regulation Annex III",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "limonene",
        "display_name": "Limonene",
        "inci_name": "LIMONENE",
        "cas": "5989-27-5",
        "chem_class": "terpene",
        "functions": ["fragrance"],
        "synonyms": ["d-limonene", "dipentene", "citrus terpenes"],
        "assertions": [
            {
                "domain": "allergen",
                "key_name": "eu26",
                "value": "declarable fragrance allergen",
                "jurisdiction": "EU",
                "source": "EU Cosmetics Regulation Annex III (fragrance allergens)",
                "evidence_grade": "A",
            },
            {
                "domain": "hazard",
                "key_name": "ghs_code",
                "value": "H317",
                "source": "ECHA CLP",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "linalool",
        "display_name": "Linalool",
        "inci_name": "LINALOOL",
        "cas": "78-70-6",
        "chem_class": "terpene alcohol",
        "functions": ["fragrance"],
        "synonyms": ["linalol", "beta-linalool"],
        "assertions": [
            {
                "domain": "allergen",
                "key_name": "eu26",
                "value": "declarable fragrance allergen",
                "jurisdiction": "EU",
                "source": "EU Cosmetics Regulation Annex III (fragrance allergens)",
                "evidence_grade": "A",
            }
        ],
    },
    {
        "chemical_id": "titanium_dioxide",
        "display_name": "Titanium Dioxide",
        "inci_name": "TITANIUM DIOXIDE",
        "cas": "13463-67-7",
        "e_number": "E171",
        "chem_class": "inorganic pigment",
        "functions": ["colourant", "uv filter"],
        "synonyms": ["ci 77891", "e171", "tio2"],
        "assertions": [
            {
                "domain": "hazard",
                "key_name": "iarc_group",
                "value": "2B",
                "source": "IARC Monographs Vol. 93 (inhalable form)",
                "evidence_grade": "B",
            },
            {
                "domain": "regulatory",
                "key_name": "banned",
                "value": "prohibited as a food additive",
                "jurisdiction": "EU",
                "product_class": "food",
                "source": "Commission Regulation (EU) 2022/63",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "aspartame",
        "display_name": "Aspartame",
        "inci_name": "ASPARTAME",
        "cas": "22839-47-0",
        "e_number": "E951",
        "chem_class": "dipeptide sweetener",
        "functions": ["sweetener"],
        "synonyms": ["e951", "aspartyl-phenylalanine methyl ester", "nutrasweet"],
        "assertions": [
            {
                "domain": "hazard",
                "key_name": "iarc_group",
                "value": "2B",
                "source": "IARC Monographs Vol. 134 (2023)",
                "evidence_grade": "B",
            },
            {
                "domain": "regulatory",
                "key_name": "restricted",
                "value": "ADI 40 mg/kg bw/day",
                "jurisdiction": "INTL",
                "product_class": "food",
                "limit_value": 40,
                "limit_unit": "mg/kg bw/day",
                "source": "JECFA / EFSA",
                "evidence_grade": "A",
            },
        ],
    },
    {
        "chemical_id": "sodium_benzoate",
        "display_name": "Sodium Benzoate",
        "inci_name": "SODIUM BENZOATE",
        "cas": "532-32-1",
        "e_number": "E211",
        "chem_class": "benzoate salt",
        "functions": ["preservative"],
        "synonyms": ["e211", "benzoate of soda", "sodium salt of benzoic acid"],
        "assertions": [
            {
                "domain": "regulatory",
                "key_name": "annex_iii",
                "value": "max 2.5% rinse-off",
                "jurisdiction": "EU",
                "product_class": "cosmetic",
                "limit_value": 2.5,
                "limit_unit": "%",
                "source": "EU Cosmetics Regulation Annex V",
                "evidence_grade": "A",
            }
        ],
    },
    {
        "chemical_id": "retinol",
        "display_name": "Retinol",
        "inci_name": "RETINOL",
        "cas": "68-26-8",
        "chem_class": "retinoid",
        "functions": ["skin conditioning"],
        "synonyms": ["vitamin a", "vitamin a1", "all-trans-retinol"],
        "assertions": [
            {
                "domain": "regulatory",
                "key_name": "annex_iii",
                "value": "max 0.3% face products",
                "jurisdiction": "EU",
                "product_class": "cosmetic",
                "limit_value": 0.3,
                "limit_unit": "%",
                "source": "Commission Regulation (EU) 2024/996",
                "evidence_grade": "A",
            }
        ],
    },
]

# arch.md 8.6: cross-reactant expansion is data, not a prompt.
CROSS_REACTANTS: list[tuple[str, str, str]] = [
    ("balsam of peru", "limonene", "moderate"),
    ("balsam of peru", "linalool", "moderate"),
    ("fragrance", "limonene", "moderate"),
    ("fragrance", "linalool", "moderate"),
    ("perfume", "limonene", "moderate"),
    ("parabens", "butylparaben", "high"),
    ("aspirin", "sodium_benzoate", "low"),
    ("vitamin a", "retinol", "high"),
]


def main() -> int:
    with session_scope() as session:
        repo = ChemicalRepository(session)

        for entry in CHEMICALS:
            repo.upsert_chemical(
                chemical_id=entry["chemical_id"],
                inci_name=entry.get("inci_name"),
                display_name=entry["display_name"],
                cas=entry.get("cas"),
                e_number=entry.get("e_number"),
                chem_class=entry.get("chem_class"),
                functions=entry.get("functions"),
                review_status="published",
            )

            # The INCI name and the display name are both surface forms people
            # (and labels) actually use.
            if entry.get("inci_name"):
                repo.add_synonym(entry["chemical_id"], entry["inci_name"], kind="inci")
            repo.add_synonym(entry["chemical_id"], entry["display_name"], kind="inci")
            if entry.get("cas"):
                repo.add_synonym(entry["chemical_id"], entry["cas"], kind="cas")
            if entry.get("e_number"):
                repo.add_synonym(entry["chemical_id"], entry["e_number"], kind="e_number")
            for synonym in entry.get("synonyms", []):
                repo.add_synonym(entry["chemical_id"], synonym, kind="synonym")

            for assertion in entry.get("assertions", []):
                repo.add_assertion(
                    entry["chemical_id"],
                    assertion.pop("domain"),
                    assertion.pop("key_name"),
                    **assertion,
                )

            print(f"  {entry['chemical_id']:<24} {len(entry.get('assertions', []))} assertion(s)")

        for rule in DEFAULT_RULES:
            repo.upsert_rule(
                rule["rule_id"],
                rule["condition"],
                rule["effect"],
                priority=rule.get("priority", 100),
                description=rule.get("description"),
                owner="seed",
            )

        for allergen, chemical_id, severity in CROSS_REACTANTS:
            repo.add_cross_reactant(allergen, chemical_id, severity, source="seed")

        stats = repo.kb_stats()

    print(f"\n{len(CHEMICALS)} chemicals, {len(DEFAULT_RULES)} rules, "
          f"{len(CROSS_REACTANTS)} cross-reactants seeded.")
    print(f"KB now holds: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
