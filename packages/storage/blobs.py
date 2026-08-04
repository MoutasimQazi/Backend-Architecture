"""Filesystem blob store.

arch.md 3 puts uploads in S3. There is no object store on this host, so blobs
live under `BLOB_DIR` (which must sit outside `public_html`) with a row in
`blob_object` for metadata. Content is addressed by sha256, so re-uploading the
same photo costs nothing and — more usefully — the OCR cache keyed on that hash
hits immediately.

The table is `blob_object`, not `blob`: BLOB is a reserved word in MySQL and
MariaDB, so the unquoted name is a syntax error in both DDL and queries.
`scripts/check_ddl.py` guards against that class of mistake returning.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.config import get_settings
from packages.guards.fetch_broker import FetchError, get_broker

logger = logging.getLogger(__name__)

_DATA_URL = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)
_BASE64_CHARS = re.compile(r"[A-Za-z0-9+/=\s]+")

# Sniffed from content rather than trusted from the client: a caller claiming
# image/png for a PHP script should not get a .php on disk.
_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
    (b"BM", "image/bmp"),
]

_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/heic": ".heic",
    "application/pdf": ".pdf",
}


def sniff_mime(raw: bytes) -> Optional[str]:
    for magic, mime in _MAGIC:
        if raw.startswith(magic):
            return mime
    # WEBP is 'RIFF....WEBP' — the marker is at offset 8, not 0.
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    # HEIC/HEIF box header at offset 4.
    if raw[4:8] == b"ftyp" and raw[8:12] in (b"heic", b"heix", b"mif1", b"heim"):
        return "image/heic"
    return None


@dataclass
class StoredBlob:
    blob_id: str
    sha256: str
    mime_type: str
    size_bytes: int
    path: Path
    deduplicated: bool = False

    def read(self) -> bytes:
        return self.path.read_bytes()


class BlobStore:
    def __init__(self, session: Session, base_dir: Optional[Path] = None) -> None:
        self.session = session
        self.settings = get_settings()
        self.base_dir = Path(base_dir or self.settings.storage.blob_dir)

    def _path_for(self, sha: str, mime: str) -> Path:
        # Two levels of fan-out: a single flat directory with 100k files makes
        # every stat() on it slow.
        suffix = _EXTENSIONS.get(mime, ".bin")
        return self.base_dir / sha[:2] / sha[2:4] / f"{sha}{suffix}"

    def put(
        self,
        raw: bytes,
        *,
        user_id: Optional[str] = None,
        declared_mime: Optional[str] = None,
        kind: str = "upload",
        parent_id: Optional[str] = None,
        page_index: Optional[int] = None,
    ) -> StoredBlob:
        if not raw:
            raise ValueError("empty blob")

        limit = self.settings.storage.max_upload_bytes
        if len(raw) > limit:
            raise ValueError(f"blob exceeds {limit} bytes")

        mime = sniff_mime(raw) or declared_mime or "application/octet-stream"
        allowed = self.settings.storage.allowed_mime
        if allowed and mime not in allowed:
            raise ValueError(f"content type '{mime}' is not accepted")

        sha = hashlib.sha256(raw).hexdigest()

        # Dedupe per owner, not globally.
        #
        # Deduping on sha256 alone handed the SECOND uploader of a given image
        # the first uploader's blob_id — a row they do not own. `get()` scopes
        # by user_id, so every later read of that handle returned nothing and
        # the scan failed with "attachment not found". Two people photographing
        # the same product is the normal case here, not an edge case.
        #
        # The file on disk stays shared (its path is content-addressed); only
        # the blob_object row is per-user. So storage is still deduplicated
        # while access stays scoped.
        existing = self.session.execute(
            text(
                "SELECT blob_id, rel_path, size_bytes FROM blob_object "
                "WHERE sha256 = :sha AND user_id <=> :uid LIMIT 1"
            ),
            {"sha": sha, "uid": user_id},
        ).mappings().first()

        if existing:
            path = self.base_dir / existing["rel_path"]
            if path.is_file():
                return StoredBlob(
                    blob_id=existing["blob_id"],
                    sha256=sha,
                    mime_type=mime,
                    size_bytes=int(existing["size_bytes"]),
                    path=path,
                    deduplicated=True,
                )
            # Row without a file: the disk was cleared behind our back. Fall
            # through and rewrite rather than hand back a path that 404s.
            logger.warning("blob row %s has no file on disk; rewriting", existing["blob_id"])

        path = self._path_for(sha, mime)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Another user may already have written these exact bytes to this exact
        # path. Identical content, so no need to rewrite it.
        if not path.is_file():
            path.write_bytes(raw)

        blob_id = uuid.uuid4().hex
        rel_path = str(path.relative_to(self.base_dir)).replace("\\", "/")
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.settings.storage.attachment_ttl_days
        )

        self.session.execute(
            text(
                """
                INSERT INTO blob_object (blob_id, user_id, sha256, mime_type, size_bytes,
                                  rel_path, kind, parent_id, page_index, expires_at)
                VALUES (:bid, :uid, :sha, :mime, :size, :path, :kind, :parent, :page, :exp)
                """
            ),
            {
                "bid": blob_id,
                "uid": user_id,
                "sha": sha,
                "mime": mime,
                "size": len(raw),
                "path": rel_path,
                "kind": kind,
                "parent": parent_id,
                "page": page_index,
                "exp": expires_at.replace(tzinfo=None),
            },
        )

        return StoredBlob(
            blob_id=blob_id, sha256=sha, mime_type=mime, size_bytes=len(raw), path=path
        )

    def get(self, blob_id: str, user_id: Optional[str] = None) -> Optional[StoredBlob]:
        clause = "AND (user_id = :uid OR user_id IS NULL)" if user_id else ""
        params: dict[str, Any] = {"bid": blob_id}
        if user_id:
            params["uid"] = user_id

        row = self.session.execute(
            text(
                f"SELECT blob_id, sha256, mime_type, size_bytes, rel_path "
                f"FROM blob_object WHERE blob_id = :bid {clause}"
            ),
            params,
        ).mappings().first()
        if not row:
            return None

        path = self.base_dir / row["rel_path"]
        if not path.is_file():
            logger.warning("blob %s missing on disk at %s", blob_id, path)
            return None

        return StoredBlob(
            blob_id=row["blob_id"],
            sha256=row["sha256"],
            mime_type=row["mime_type"],
            size_bytes=int(row["size_bytes"]),
            path=path,
        )

    def purge_expired(self, limit: int = 500) -> int:
        rows = self.session.execute(
            text(
                "SELECT blob_id, rel_path FROM blob_object "
                "WHERE expires_at IS NOT NULL AND expires_at <= UTC_TIMESTAMP(3) LIMIT :lim"
            ),
            {"lim": limit},
        ).mappings().all()

        removed = 0
        for row in rows:
            path = self.base_dir / row["rel_path"]
            try:
                if path.is_file():
                    path.unlink()
            except OSError as exc:
                logger.warning("could not delete blob file %s: %s", path, exc)
                continue
            self.session.execute(
                text("DELETE FROM blob_object WHERE blob_id = :bid"), {"bid": row["blob_id"]}
            )
            removed += 1
        return removed


# -- attachment intake ------------------------------------------------------


_BASE64_HINT = (
    "send the image as a multipart/form-data file part named 'file' instead "
    "of base64"
)


def decode_attachment_payload(value: str) -> tuple[bytes, Optional[str]]:
    """Resolve an http(s) attachment reference to bytes.

    Base64 intake was removed: it inflated every image by a third, forced the
    whole file through the JSON body limit, and meant the bytes were parsed
    twice before anything could validate them. Images now arrive as raw
    multipart file parts and never pass through this function at all.

    What remains is the URL case — the one that used to be SSRF, because the
    old `app.py` called `urlopen` on caller-supplied URLs directly. It goes
    through the fetch broker, which resolves DNS and re-checks every address
    before connecting.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty attachment payload")

    value = value.strip()

    if _DATA_URL.match(value):
        raise ValueError(f"data: URLs are no longer accepted — {_BASE64_HINT}")

    if value.startswith(("http://", "https://")):
        try:
            result = get_broker().get(value)
        except FetchError as exc:
            raise ValueError(f"could not fetch attachment: {exc}") from exc
        if result.status_code != 200:
            raise ValueError(f"attachment URL returned {result.status_code}")
        return result.content, result.headers.get("content-type", "").split(";")[0] or None

    # A bare base64 blob is the most likely thing an un-migrated client sends,
    # and "not a valid URL" would send them looking in the wrong place.
    if _looks_like_base64(value):
        raise ValueError(f"base64 attachments are no longer accepted — {_BASE64_HINT}")

    raise ValueError(f"attachment must be an http(s) URL — {_BASE64_HINT}")


def _looks_like_base64(value: str) -> bool:
    if len(value) < 64:
        return False
    sample = value[:256]
    return bool(_BASE64_CHARS.fullmatch(sample))
