"""User-scoped tools (arch.md 6.2).

Every tool is constructed with the caller's `user_id` bound at build time. A
tool physically cannot address another user's rows — tenancy is a repository
predicate, not a prompt instruction.
"""

from packages.tools.health_tools import HealthToolset, TOOL_SCHEMAS

__all__ = ["HealthToolset", "TOOL_SCHEMAS"]
