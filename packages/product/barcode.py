"""Barcode decoding and validation.

arch.md 8.2 says to keep the current product-format gate — that logic is sound —
and add check-digit validation, GTIN normalisation and multi-frame voting. That
is what this module is: the retail-symbology gate from the old `app.py` carried
over intact, with the three additions layered on.

cv2 and zxingcpp are imported lazily. They are the heaviest wheels in the stack
and the most likely to be missing on a shared host; a deploy without them must
degrade to "barcode scanning unavailable, photograph the panel instead" rather
than fail to boot.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Only retail/product symbologies may route an upload down the barcode branch.
# A QR code or DataMatrix on a lab report, invoice or shipping label is not a
# product identifier, and treating it as one used to hijack the whole request.
PRODUCT_BARCODE_FORMATS = {
    "EAN13",
    "EAN8",
    "UPCA",
    "UPCE",
    "ITF",
    "Code128",
    "Code39",
    "Code93",
}
_PRODUCT_FORMAT_PREFIXES = ("DataBar",)

# GS1 payloads carry application identifiers — DataBar and GS1-128 decode as
# "(01)03614141000125", where AI 01 is the GTIN.
_GS1_GTIN_RE = re.compile(r"\(01\)(\d{14})")

_cv2: Any = None
_zxing: Any = None
_backend_checked = False
_backend_error: Optional[str] = None


def _load_backend() -> bool:
    global _cv2, _zxing, _backend_checked, _backend_error
    if _backend_checked:
        return _zxing is not None

    _backend_checked = True
    try:
        import cv2  # type: ignore[import-not-found]
        import zxingcpp  # type: ignore[import-not-found]

        _cv2, _zxing = cv2, zxingcpp
        return True
    except ImportError as exc:
        _backend_error = str(exc)
        logger.warning("barcode backend unavailable (%s); scanning is disabled", exc)
        return False


def backend_available() -> bool:
    return _load_backend()


def backend_error() -> Optional[str]:
    _load_backend()
    return _backend_error


@dataclass
class BarcodeCandidate:
    payload: str
    symbology: str
    is_product: bool
    check_digit_valid: Optional[bool] = None
    gtin14: Optional[str] = None


@dataclass
class BarcodeResult:
    barcode: Optional[str] = None
    gtin14: Optional[str] = None
    symbology: Optional[str] = None
    confidence: float = 0.0
    frames_agreeing: int = 0
    frames_scanned: int = 0
    rejected_symbols: list[str] = field(default_factory=list)
    candidates: list[BarcodeCandidate] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.barcode is not None

    @property
    def saw_non_retail_symbol(self) -> bool:
        """A code was visible but is not a product barcode — a QR link or an
        asset tag. Saying so beats "your photo was unreadable" (arch.md 8.7)."""
        return not self.found and bool(self.rejected_symbols)


def format_name(fmt: Any) -> str:
    """Normalise zxingcpp's format value to a compact name.

    zxingcpp reports 'Code 128', 'EAN-13', 'QR Code' — with spaces and hyphens.
    Stripping only '-' and '_' left 'Code 128', which matched nothing in
    PRODUCT_BARCODE_FORMATS, so every Code128/Code39/DataBar scan was silently
    discarded no matter what it encoded.
    """
    name = str(fmt).split(".")[-1]
    return re.sub(r"[\s_-]+", "", name).strip()


def is_product_format(fmt_name: str) -> bool:
    if fmt_name in PRODUCT_BARCODE_FORMATS:
        return True
    return fmt_name.startswith(_PRODUCT_FORMAT_PREFIXES)


def normalise_payload(text_value: str) -> str:
    text_value = (text_value or "").strip()
    match = _GS1_GTIN_RE.search(text_value)
    if match:
        return match.group(1)
    if len(text_value) == 16 and text_value.isdigit() and text_value.startswith("01"):
        return text_value[2:]  # unparenthesised GS1 element string
    return text_value


def gtin_check_digit(digits: str) -> Optional[int]:
    """GS1 mod-10. Weights alternate 3,1 from the right, excluding the check."""
    body = digits[:-1]
    if not body.isdigit():
        return None
    total = 0
    for index, char in enumerate(reversed(body)):
        weight = 3 if index % 2 == 0 else 1
        total += int(char) * weight
    return (10 - (total % 10)) % 10


def validate_check_digit(digits: str) -> Optional[bool]:
    """True/False for GTIN-8/12/13/14; None when the length is not a GTIN.

    Code128/Code39 carry arbitrary numeric article numbers that are not GTINs,
    so "not a GTIN length" must stay distinct from "bad check digit" — the
    former is still usable, the latter means a misread.
    """
    if not digits.isdigit() or len(digits) not in (8, 12, 13, 14):
        return None
    expected = gtin_check_digit(digits)
    if expected is None:
        return None
    return expected == int(digits[-1])


def to_gtin14(digits: str) -> Optional[str]:
    """Left-pad to 14 so EAN-13, UPC-A and GTIN-14 for the same article all
    resolve to one key. Product databases are keyed inconsistently, and this is
    what makes the local `product` table hit."""
    if not digits.isdigit() or len(digits) > 14:
        return None
    return digits.rjust(14, "0")


def expand_upce(payload: str) -> Optional[str]:
    """UPC-E is a zero-suppressed UPC-A. Product databases index the expanded
    form, so a UPC-E scan misses every lookup unless it is expanded first."""
    if not payload.isdigit() or len(payload) != 8:
        return None

    number_system, body, check = payload[0], payload[1:7], payload[7]
    if number_system not in ("0", "1"):
        return None

    last = body[5]
    if last in ("0", "1", "2"):
        expanded = body[:2] + last + "0000" + body[2:5]
    elif last == "3":
        expanded = body[:3] + "00000" + body[3:5]
    elif last == "4":
        expanded = body[:4] + "00000" + body[4]
    else:
        expanded = body[:5] + "0000" + last

    return number_system + expanded + check


def is_product_barcode(payload: str, fmt_name: str) -> bool:
    """A linear symbology carrying a plausible article number.

    2D codes (QR, DataMatrix, Aztec, PDF417) are excluded: on a document they
    are a link or a record id, not a product. Among linear codes the payload
    must be all digits — that is what rejects 'PATIENT-99213' style Code128
    asset tags while still accepting numeric article numbers.
    """
    if not is_product_format(fmt_name):
        return False
    digits = normalise_payload(payload)
    return digits.isdigit() and 6 <= len(digits) <= 14


def _to_cv_image(image_input: Any) -> Any:
    import numpy as np

    if not _load_backend():
        raise RuntimeError("barcode backend unavailable")

    if isinstance(image_input, (bytes, bytearray)):
        buffer = np.frombuffer(bytes(image_input), np.uint8)
        return _cv2.imdecode(buffer, _cv2.IMREAD_COLOR)
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    raise ValueError("unsupported image input")


def decode_frame(image_input: Any) -> list[BarcodeCandidate]:
    """Decode one image. Never raises — one bad frame must not abort a batch."""
    if not _load_backend():
        return []

    try:
        img = _to_cv_image(image_input)
    except (ValueError, RuntimeError) as exc:
        logger.warning("decode_frame: %s", exc)
        return []

    if img is None:
        logger.warning("decode_frame: image could not be decoded")
        return []

    raw: list[Any] = []
    try:
        raw.extend(_zxing.read_barcodes(img))
        if not raw:
            gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
            raw.extend(_zxing.read_barcodes(gray))
            if not raw:
                _, thresh = _cv2.threshold(
                    gray, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU
                )
                raw.extend(_zxing.read_barcodes(thresh))
    except Exception as exc:
        logger.warning("decode_frame: decode failed: %s", exc)
        return []

    candidates: list[BarcodeCandidate] = []
    for obj in raw:
        fmt = format_name(obj.format)
        payload = (obj.text or "").strip()
        if not payload:
            continue

        product = is_product_barcode(payload, fmt)
        digits = normalise_payload(payload) if product else payload

        if product and fmt == "UPCE":
            expanded = expand_upce(digits)
            if expanded:
                digits = expanded

        candidates.append(
            BarcodeCandidate(
                payload=digits,
                symbology=fmt,
                is_product=product,
                check_digit_valid=validate_check_digit(digits) if product else None,
                gtin14=to_gtin14(digits) if product else None,
            )
        )

    return candidates


def scan(images: list[bytes]) -> BarcodeResult:
    """Multi-frame voting (arch.md 8.2).

    A scanner sends several frames of the same pack. One frame can misread a
    digit and still produce a valid-looking code; agreement across frames is
    what separates a real read from a plausible one. A payload whose check
    digit fails is never returned — that is a misread, not a product.
    """
    result = BarcodeResult(frames_scanned=len(images))
    votes: Counter[str] = Counter()
    by_payload: dict[str, BarcodeCandidate] = {}
    rejected: set[str] = set()

    for raw in images:
        for candidate in decode_frame(raw):
            result.candidates.append(candidate)
            if not candidate.is_product:
                rejected.add(candidate.symbology)
                continue
            if candidate.check_digit_valid is False:
                logger.info(
                    "rejecting %s: check digit failed (misread)", candidate.payload
                )
                continue
            votes[candidate.payload] += 1
            by_payload.setdefault(candidate.payload, candidate)

    result.rejected_symbols = sorted(rejected)

    if not votes:
        return result

    payload, count = votes.most_common(1)[0]
    winner = by_payload[payload]

    result.barcode = payload
    result.gtin14 = winner.gtin14
    result.symbology = winner.symbology
    result.frames_agreeing = count

    # Agreement across frames and a verified check digit are independent
    # signals; a single frame with a valid GTIN check digit is already strong.
    confidence = 0.5
    if winner.check_digit_valid is True:
        confidence += 0.35
    if count >= 2:
        confidence += 0.15
    result.confidence = round(min(confidence, 1.0), 2)

    return result
