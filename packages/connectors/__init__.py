"""Upstream connectors. Every one of them fetches through the broker."""

from packages.connectors.openfacts import OpenFactsConnector, ProductRecord

__all__ = ["OpenFactsConnector", "ProductRecord"]
