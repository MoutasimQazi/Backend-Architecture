"""Offline ETL. Fills the Chemical KB and the evidence store on a schedule.

arch.md 8.4: "A scan at runtime touches zero external toxicology APIs." This is
where those APIs are called instead — in a job, reviewable and versioned.
"""

from packages.etl.chemical import ChemicalEtl, EtlOutcome

__all__ = ["ChemicalEtl", "EtlOutcome"]
