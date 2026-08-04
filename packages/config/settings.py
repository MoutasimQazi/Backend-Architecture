"""Environment-driven settings.

Deployment target is cPanel shared hosting: MySQL/MariaDB only, no Redis, no
Docker, no object store. Everything that arch.md assigns to Redis or S3 is
mapped onto MySQL or the local filesystem here, behind the same interfaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

try:  # optional: present in dev, absent on a bare cPanel Python
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not a dependency
    pass


ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = field(default_factory=lambda: _env("DB_HOST", "213.136.64.26"))
    port: int = field(default_factory=lambda: _env_int("DB_PORT", 3306))
    user: str = field(default_factory=lambda: _env("DB_USER", "movenetics_fitness"))
    password: str = field(default_factory=lambda: _env("DB_PASSWORD", ""))
    name: str = field(default_factory=lambda: _env("DB_NAME", "movenetics_fitness"))
    charset: str = field(default_factory=lambda: _env("DB_CHARSET", "utf8mb4"))

    # Shared hosting caps concurrent connections hard. Keep the pool small and
    # recycle before the server's wait_timeout drops the socket underneath us.
    pool_size: int = field(default_factory=lambda: _env_int("DB_POOL_SIZE", 5))
    max_overflow: int = field(default_factory=lambda: _env_int("DB_MAX_OVERFLOW", 5))
    pool_recycle: int = field(default_factory=lambda: _env_int("DB_POOL_RECYCLE", 280))
    pool_timeout: int = field(default_factory=lambda: _env_int("DB_POOL_TIMEOUT", 30))
    echo: bool = field(default_factory=lambda: _env_bool("DB_ECHO", False))

    @property
    def url(self) -> str:
        """SQLAlchemy URL. PyMySQL is pure-Python, so it installs without a
        compiler — which a shared host usually does not give you."""
        return (
            f"mysql+pymysql://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.name}?charset={self.charset}"
        )

    @property
    def url_safe(self) -> str:
        """Same URL with the password masked, for logs."""
        return (
            f"mysql+pymysql://{self.user}:***"
            f"@{self.host}:{self.port}/{self.name}?charset={self.charset}"
        )


@dataclass(frozen=True)
class CacheSettings:
    l1_max_entries: int = field(default_factory=lambda: _env_int("CACHE_L1_MAX_ENTRIES", 1024))
    l1_ttl_seconds: int = field(default_factory=lambda: _env_int("CACHE_L1_TTL_SECONDS", 60))
    enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))

    # L3 semantic-hit floor. arch.md 7.3: an L3 hit is never returned blind.
    semantic_threshold: float = field(
        default_factory=lambda: _env_float("CACHE_SEMANTIC_THRESHOLD", 0.93)
    )
    negative_ttl_seconds: int = field(
        default_factory=lambda: _env_int("CACHE_NEGATIVE_TTL_SECONDS", 60)
    )

    # arch.md 7.3 TTL-by-category, in seconds.
    ttl_by_category: dict[str, int] = field(
        default_factory=lambda: {
            "AppSupport": _env_int("CACHE_TTL_APPSUPPORT", 30 * 24 * 3600),
            "General": _env_int("CACHE_TTL_GENERAL", 14 * 24 * 3600),
            "Nutrition": _env_int("CACHE_TTL_NUTRITION", 24 * 3600),
            "Workout": _env_int("CACHE_TTL_WORKOUT", 24 * 3600),
            "Product": _env_int("CACHE_TTL_PRODUCT", 7 * 24 * 3600),
            "Research": _env_int("CACHE_TTL_RESEARCH", 7 * 24 * 3600),
            "Medical": _env_int("CACHE_TTL_MEDICAL", 3600),
        }
    )

    def ttl_for(self, category: str | None) -> int:
        if not category:
            return self.ttl_by_category["General"]
        return self.ttl_by_category.get(category, self.ttl_by_category["General"])


@dataclass(frozen=True)
class RouterSettings:
    """arch.md 5.3: thresholds are config, not constants."""

    tau_faq: float = field(default_factory=lambda: _env_float("ROUTER_TAU_FAQ", 0.82))
    tau_low: float = field(default_factory=lambda: _env_float("ROUTER_TAU_LOW", 0.45))
    tau_cache: float = field(default_factory=lambda: _env_float("ROUTER_TAU_CACHE", 0.93))

    # Per-category overrides: Medical wants precision, AppSupport wants recall.
    tau_faq_by_category: dict[str, float] = field(
        default_factory=lambda: {
            "Medical": _env_float("ROUTER_TAU_FAQ_MEDICAL", 0.90),
            "AppSupport": _env_float("ROUTER_TAU_FAQ_APPSUPPORT", 0.74),
        }
    )

    def tau_faq_for(self, category: str | None) -> float:
        if not category:
            return self.tau_faq
        return self.tau_faq_by_category.get(category, self.tau_faq)


@dataclass(frozen=True)
class ModelSettings:
    """arch.md 13: small models do small jobs; only synthesis uses the large one."""

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY", ""))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))

    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    hf_base_url: str = field(
        default_factory=lambda: _env("MICROSOFT_BASE_URL", "https://router.huggingface.co/v1")
    )

    small_model: str = field(default_factory=lambda: _env("SMALL_MODEL", "gpt-4o-mini"))
    large_model: str = field(default_factory=lambda: _env("LARGE_MODEL", "gpt-4o"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 1536))

    request_timeout: int = field(default_factory=lambda: _env_int("MODEL_TIMEOUT", 60))
    max_retries: int = field(default_factory=lambda: _env_int("MODEL_MAX_RETRIES", 2))

    @property
    def any_configured(self) -> bool:
        return bool(self.openai_api_key or self.deepseek_api_key or self.hf_token)


@dataclass(frozen=True)
class JobSettings:
    """DB-backed queue settings. No broker on this host."""

    poll_interval_seconds: float = field(
        default_factory=lambda: _env_float("JOB_POLL_INTERVAL", 3.0)
    )
    batch_size: int = field(default_factory=lambda: _env_int("JOB_BATCH_SIZE", 5))
    max_attempts: int = field(default_factory=lambda: _env_int("JOB_MAX_ATTEMPTS", 3))
    lease_seconds: int = field(default_factory=lambda: _env_int("JOB_LEASE_SECONDS", 600))
    result_ttl_seconds: int = field(
        default_factory=lambda: _env_int("JOB_RESULT_TTL_SECONDS", 7 * 24 * 3600)
    )
    # A cPanel cron tick should exit rather than run forever.
    max_runtime_seconds: int = field(
        default_factory=lambda: _env_int("JOB_MAX_RUNTIME_SECONDS", 0)
    )


@dataclass(frozen=True)
class StorageSettings:
    """No S3 on this host: blobs go to a filesystem directory outside webroot."""

    blob_dir: Path = field(
        default_factory=lambda: Path(_env("BLOB_DIR", str(ROOT / "var" / "blobs")))
    )
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024)
    )
    max_attachments: int = field(default_factory=lambda: _env_int("MAX_ATTACHMENTS", 5))
    allowed_mime: list[str] = field(
        default_factory=lambda: _env_list(
            "ALLOWED_MIME", "image/jpeg,image/png,image/webp,image/heic,application/pdf"
        )
    )
    attachment_ttl_days: int = field(default_factory=lambda: _env_int("ATTACHMENT_TTL_DAYS", 30))


@dataclass(frozen=True)
class SecuritySettings:
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", ""))
    jwt_algorithm: str = field(default_factory=lambda: _env("JWT_ALGORITHM", "HS256"))
    jwt_audience: str = field(default_factory=lambda: _env("JWT_AUDIENCE", "fitness-api"))
    # Dev escape hatch: trust an X-User-Id header. Never enable in production.
    allow_header_auth: bool = field(default_factory=lambda: _env_bool("ALLOW_HEADER_AUTH", False))

    cors_origins: list[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", "*"))

    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MIN", 30))
    rate_limit_per_day: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_DAY", 500))

    # arch.md 13: the fetch broker's allowlist. Empty means "deny all outbound
    # user-supplied URLs", which is the safe default.
    fetch_allowlist: list[str] = field(
        default_factory=lambda: _env_list(
            "FETCH_ALLOWLIST",
            "world.openfoodfacts.org,world.openbeautyfacts.org,world.openproductsfacts.org,"
            "pubchem.ncbi.nlm.nih.gov,eutils.ncbi.nlm.nih.gov,www.ebi.ac.uk,api.crossref.org,"
            "api.openalex.org,api.semanticscholar.org",
        )
    )
    fetch_timeout_seconds: int = field(default_factory=lambda: _env_int("FETCH_TIMEOUT", 20))
    fetch_max_bytes: int = field(default_factory=lambda: _env_int("FETCH_MAX_BYTES", 5 * 1024 * 1024))


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("APP_ENV", "production"))
    debug: bool = field(default_factory=lambda: _env_bool("APP_DEBUG", False))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    root: Path = ROOT

    # Bumping either invalidates every cache entry that used them (arch.md 7.2).
    prompt_version: str = field(default_factory=lambda: _env("PROMPT_VERSION", "v1"))
    kb_version: str = field(default_factory=lambda: _env("KB_VERSION", "v1"))

    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    router: RouterSettings = field(default_factory=RouterSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    jobs: JobSettings = field(default_factory=JobSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)

    def validate(self) -> list[str]:
        """Return a list of configuration problems. Empty means good to go."""
        problems: list[str] = []
        if not self.db.password:
            problems.append("DB_PASSWORD is not set")
        if not self.db.host:
            problems.append("DB_HOST is not set")
        if not self.models.any_configured:
            problems.append("no model provider key set (OPENAI_API_KEY / DEEPSEEK_API_KEY / HF_TOKEN)")
        if self.env == "production":
            if not self.security.jwt_secret:
                problems.append("JWT_SECRET is not set (required outside development)")
            if self.security.allow_header_auth:
                problems.append("ALLOW_HEADER_AUTH must be off in production")
            if self.security.cors_origins == ["*"]:
                problems.append("CORS_ORIGINS is '*' in production")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
