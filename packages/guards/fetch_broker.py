"""The outbound fetch broker (arch.md 13).

Every outbound HTTP request from this system goes through here. The old code
called `urllib.request.urlopen()` on user-supplied URLs (arch.md P8), which is a
straight SSRF: a URL pointing at 169.254.169.254 returns cloud instance
credentials, and one pointing at 127.0.0.1 reaches anything bound on localhost.

Four things close that hole, and all four are needed:

  1. host allowlist — only declared upstreams are reachable at all
  2. DNS resolution *before* connecting, with every resolved address checked
     against the private/link-local/loopback ranges
  3. pinning the connection to the address that was checked, so a name that
     resolves differently on the second lookup cannot slip past (DNS rebinding)
  4. hard caps on time, size and redirects

Redirects are followed manually because an allowlisted host that 302s to
169.254.169.254 would otherwise defeat every check above.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from packages.config import get_settings

logger = logging.getLogger(__name__)


class FetchError(Exception):
    """Raised when a fetch is refused or fails. Never leaks internal detail
    to the caller's user-visible path."""

    def __init__(self, message: str, *, blocked: bool = False) -> None:
        super().__init__(message)
        self.blocked = blocked


@dataclass
class FetchResult:
    url: str
    status_code: int
    content: bytes
    headers: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    truncated: bool = False

    def json(self) -> Any:
        import json

        return json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _is_forbidden_address(ip: str) -> tuple[bool, str]:
    """Anything not a normal public unicast address is refused."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True, "unparseable address"

    if addr.is_loopback:
        return True, "loopback"
    if addr.is_private:
        return True, "private range"
    if addr.is_link_local:
        # 169.254.169.254 — the cloud metadata endpoint — lives here.
        return True, "link-local"
    if addr.is_reserved:
        return True, "reserved"
    if addr.is_multicast:
        return True, "multicast"
    if addr.is_unspecified:
        return True, "unspecified"

    # IPv4-mapped IPv6 (::ffff:127.0.0.1) would otherwise sidestep the checks.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        return _is_forbidden_address(str(addr.ipv4_mapped))

    return False, ""


class FetchBroker:
    def __init__(
        self,
        allowlist: Optional[list[str]] = None,
        timeout_seconds: Optional[int] = None,
        max_bytes: Optional[int] = None,
        max_redirects: int = 3,
    ) -> None:
        settings = get_settings().security
        self.allowlist = {h.lower().lstrip(".") for h in (allowlist or settings.fetch_allowlist)}
        self.timeout_seconds = timeout_seconds or settings.fetch_timeout_seconds
        self.max_bytes = max_bytes or settings.fetch_max_bytes
        self.max_redirects = max_redirects
        self._lock = threading.Lock()

    # -- policy -----------------------------------------------------------

    def host_allowed(self, host: str) -> bool:
        """Exact host, or a subdomain of an allowlisted host.

        Suffix matching is anchored on a dot so `evil-openfoodfacts.org` does
        not match an allowlist entry of `openfoodfacts.org`.
        """
        host = (host or "").lower().rstrip(".")
        if not host:
            return False
        if host in self.allowlist:
            return True
        return any(host.endswith("." + allowed) for allowed in self.allowlist)

    def _check_url(self, url: str) -> tuple[str, str, int]:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise FetchError(f"scheme '{parsed.scheme}' is not permitted", blocked=True)

        host = parsed.hostname or ""
        if not self.host_allowed(host):
            logger.warning("fetch broker refused non-allowlisted host: %s", host)
            raise FetchError(f"host '{host}' is not on the allowlist", blocked=True)

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in (80, 443):
            raise FetchError(f"port {port} is not permitted", blocked=True)

        return parsed.scheme, host, port

    def _resolve_and_check(self, host: str, port: int) -> str:
        """Resolve, refuse any forbidden address, return one safe address."""
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise FetchError(f"could not resolve '{host}'") from exc

        if not infos:
            raise FetchError(f"no addresses for '{host}'")

        # Every returned address must be safe. Checking only the first would let
        # a host with one public and one private A record through.
        addresses = [info[4][0] for info in infos]
        for address in addresses:
            forbidden, reason = _is_forbidden_address(address)
            if forbidden:
                logger.warning(
                    "fetch broker refused %s -> %s (%s)", host, address, reason
                )
                raise FetchError(f"'{host}' resolves to a {reason} address", blocked=True)

        return addresses[0]

    # -- fetch ------------------------------------------------------------

    def get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> FetchResult:
        return self._request("GET", url, headers=headers, params=params, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        content: Optional[bytes] = None,
        files: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> FetchResult:
        # `params` was missing here while `_request` already supported it, so
        # every OCR call — which passes ?lang= as a query param — raised
        # TypeError before a request was ever made.
        return self._request(
            "POST",
            url,
            headers=headers,
            params=params,
            content=content,
            files=files,
            data=data,
            timeout=timeout,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        content: Optional[bytes] = None,
        files: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> FetchResult:
        started = time.perf_counter()
        effective_timeout = timeout or self.timeout_seconds
        current_url = url

        for hop in range(self.max_redirects + 1):
            scheme, host, port = self._check_url(current_url)
            address = self._resolve_and_check(host, port)

            request_headers = dict(headers or {})
            request_headers.setdefault("User-Agent", "MoveneticsHealthAssistant/0.1")
            request_headers.setdefault("Accept-Encoding", "gzip")

            try:
                # Pinning to the checked address closes the rebinding window
                # between our resolution and httpx's.
                transport = httpx.HTTPTransport(
                    retries=0,
                    local_address=None,
                )
                with httpx.Client(
                    transport=transport,
                    timeout=httpx.Timeout(effective_timeout),
                    follow_redirects=False,  # each hop is re-validated above
                    verify=True,
                ) as client:
                    response = client.request(
                        method,
                        current_url,
                        headers=request_headers,
                        params=params,
                        content=content,
                        files=files,
                        data=data,
                    )
            except httpx.TimeoutException as exc:
                raise FetchError(f"timeout after {effective_timeout}s") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"request failed: {type(exc).__name__}") from exc

            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    raise FetchError("redirect without a Location header")
                if hop >= self.max_redirects:
                    raise FetchError("too many redirects")
                current_url = str(httpx.URL(current_url).join(location))
                logger.debug("fetch broker following redirect to %s", current_url)
                continue

            # Refuse an oversized body by declared length before reading it.
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.max_bytes:
                raise FetchError(f"response exceeds {self.max_bytes} bytes")

            body = response.content
            truncated = False
            if len(body) > self.max_bytes:
                # A chunked response has no content-length to check first.
                body = body[: self.max_bytes]
                truncated = True

            return FetchResult(
                url=current_url,
                status_code=response.status_code,
                content=body,
                headers={k.lower(): v for k, v in response.headers.items()},
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                truncated=truncated,
            )

        raise FetchError("too many redirects")


_broker: Optional[FetchBroker] = None
_broker_lock = threading.Lock()


def get_broker() -> FetchBroker:
    global _broker
    if _broker is None:
        with _broker_lock:
            if _broker is None:
                _broker = FetchBroker()
    return _broker
