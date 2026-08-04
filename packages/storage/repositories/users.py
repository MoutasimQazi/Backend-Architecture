"""User, consent and profile access."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.domain.enums import ConsentScope
from packages.domain.models import ConsentState, Profile


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def exists(self, user_id: str) -> bool:
        row = self.session.execute(
            text("SELECT 1 FROM app_user WHERE user_id = :uid"), {"uid": user_id}
        ).first()
        return row is not None

    def get(self, user_id: str) -> Optional[dict[str, Any]]:
        row = self.session.execute(
            text(
                "SELECT user_id, external_id, email, locale, jurisdiction, status, created_at "
                "FROM app_user WHERE user_id = :uid"
            ),
            {"uid": user_id},
        ).mappings().first()
        return dict(row) if row else None

    def upsert(
        self,
        user_id: str,
        external_id: Optional[str] = None,
        email: Optional[str] = None,
        locale: str = "en",
        jurisdiction: str = "IN",
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO app_user (user_id, external_id, email, locale, jurisdiction)
                VALUES (:uid, :ext, :email, :locale, :jur)
                ON DUPLICATE KEY UPDATE
                    external_id = COALESCE(VALUES(external_id), external_id),
                    email       = COALESCE(VALUES(email), email),
                    locale      = VALUES(locale),
                    jurisdiction = VALUES(jurisdiction)
                """
            ),
            {
                "uid": user_id,
                "ext": external_id,
                "email": email,
                "locale": locale,
                "jur": jurisdiction,
            },
        )

    # -- consent ----------------------------------------------------------

    def get_consent(self, user_id: str) -> ConsentState:
        rows = self.session.execute(
            text(
                "SELECT scope FROM consent_scope "
                "WHERE user_id = :uid AND granted = 1 AND revoked_at IS NULL"
            ),
            {"uid": user_id},
        ).all()
        scopes: list[ConsentScope] = []
        for (scope,) in rows:
            try:
                scopes.append(ConsentScope(scope))
            except ValueError:
                continue  # a scope the code no longer knows about is simply not granted
        return ConsentState(granted_scopes=scopes)

    def grant(self, user_id: str, scope: ConsentScope) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO consent_scope (user_id, scope, granted, revoked_at)
                VALUES (:uid, :scope, 1, NULL)
                ON DUPLICATE KEY UPDATE granted = 1, revoked_at = NULL,
                                        granted_at = CURRENT_TIMESTAMP(3)
                """
            ),
            {"uid": user_id, "scope": str(scope.value)},
        )

    def revoke(self, user_id: str, scope: ConsentScope) -> None:
        self.session.execute(
            text(
                "UPDATE consent_scope SET granted = 0, revoked_at = CURRENT_TIMESTAMP(3) "
                "WHERE user_id = :uid AND scope = :scope"
            ),
            {"uid": user_id, "scope": str(scope.value)},
        )

    # -- profile ----------------------------------------------------------

    def get_profile(self, user_id: str) -> tuple[Profile, Optional[int]]:
        """Returns the profile and its version (the cache fingerprint input)."""
        row = self.session.execute(
            text(
                "SELECT display_name, age_band, sex, height_cm, weight_kg, pregnancy_status, "
                "       goals, preferences, version "
                "FROM user_profile WHERE user_id = :uid"
            ),
            {"uid": user_id},
        ).mappings().first()

        if not row:
            return Profile(), None

        return (
            Profile(
                display_name=row["display_name"],
                age_band=row["age_band"],
                sex=row["sex"],
                height_cm=float(row["height_cm"]) if row["height_cm"] is not None else None,
                weight_kg=float(row["weight_kg"]) if row["weight_kg"] is not None else None,
                pregnancy_status=row["pregnancy_status"],
                goals=_loads(row["goals"]) or [],
                preferences=_loads(row["preferences"]) or [],
            ),
            int(row["version"]),
        )

    def upsert_profile(self, user_id: str, **fields: Any) -> None:
        allowed = {
            "display_name",
            "date_of_birth",
            "age_band",
            "sex",
            "height_cm",
            "weight_kg",
            "pregnancy_status",
        }
        json_fields = {"goals", "preferences"}

        params: dict[str, Any] = {"uid": user_id}
        columns: list[str] = []

        for key, value in fields.items():
            if key in allowed:
                columns.append(key)
                params[key] = value
            elif key in json_fields:
                columns.append(key)
                params[key] = json.dumps(value or [])

        if not columns:
            return

        col_sql = ", ".join(columns)
        val_sql = ", ".join(f":{c}" for c in columns)
        # Every write bumps `version`, which is what busts this user's caches.
        update_sql = ", ".join(f"{c} = VALUES({c})" for c in columns)

        self.session.execute(
            text(
                f"""
                INSERT INTO user_profile (user_id, {col_sql}, version)
                VALUES (:uid, {val_sql}, 1)
                ON DUPLICATE KEY UPDATE {update_sql}, version = version + 1
                """
            ),
            params,
        )
