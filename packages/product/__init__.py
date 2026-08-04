"""Product analysis: barcode, OCR, ingredient resolution, hazard rules.

The runtime path touches zero external toxicology APIs (arch.md 8.4) — hazard
data comes from the curated Chemical KB, and unknown ingredients become jobs.
"""

from packages.product.analyzer import AnalysisTrace, ProductAnalyzer
from packages.product.rules import HazardRulesEngine

__all__ = ["AnalysisTrace", "HazardRulesEngine", "ProductAnalyzer"]
