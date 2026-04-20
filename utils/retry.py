"""Rate-limit retry primitives used by every CLI-backed interface.

Each provider (Anthropic, OpenAI, Google) has its own 429/rate-limit
error shape, surfaced via the CLI subprocess's stderr. A per-provider
``RateLimitDetector`` classifies a failed subprocess result as a
rate-limit hit (or not); if it is, ``with_rate_limit_retry`` sleeps and
retries with exponential backoff + jitter, honoring a parsed
``Retry-After`` hint when present. Exhaustion raises
``RateLimitExhausted`` so the worker can mark the instance with
``error_kind=rate_limit_exhausted``.

Detector classification is string-matching the provider's error lingo —
brittle by design. If a provider changes its wording, we'll miss the
class and surface a plain ``subprocess_error``; that's the documented
fallback (see docs/phase-4-spec.md §7.1).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


@dataclass(frozen=True)
class RateLimitHit:
    """A single classified rate-limit failure."""

    retry_after_s: Optional[float]
    detail: str


@dataclass(frozen=True)
class RateLimitEvent:
    """Emitted via the ``on_retry`` callback before each sleep."""

    interface: str
    attempt: int  # 1-based — 1 = first failure, 2 = second, etc.
    wait_seconds: float
    detail: str


@dataclass
class RateLimitPolicy:
    """Per-instance retry policy. Defaults per docs/phase-4-spec.md §7.1."""

    max_retries: int = 5
    base_seconds: float = 10.0
    max_seconds: float = 160.0
    jitter_seconds: float = 2.0


class RateLimitDetector(Protocol):
    """Classify a failed subprocess result as a rate-limit hit or not."""

    def classify(self, result: dict) -> Optional[RateLimitHit]: ...


class RateLimitExhausted(RuntimeError):
    """Raised when ``max_retries + 1`` consecutive 429s land for one instance."""

    def __init__(self, interface: str, attempts: int, detail: str) -> None:
        self.interface = interface
        self.attempts = attempts
        self.detail = detail
        super().__init__(
            f"{interface} rate limit not cleared after {attempts} attempts: {detail}"
        )


# ---------- Provider detectors -------------------------------------------------


# Anthropic API surfaces ``rate_limit_error`` and ``overloaded_error`` in the
# error body; the claude CLI forwards that onto stderr. We also treat plain
# ``429`` as a rate-limit signal for defence-in-depth.
_CLAUDE_PATTERNS = (
    re.compile(r"rate_limit_error", re.I),
    re.compile(r"overloaded_error", re.I),
    re.compile(r"\b429\b"),
)
# Retry-After header value (seconds) — both ``Retry-After: 30`` and
# ``retry-after: 30`` are accepted.
_RETRY_AFTER = re.compile(r"retry-after[:\s]+(\d+(?:\.\d+)?)", re.I)


class ClaudeRateLimitDetector:
    def classify(self, result: dict) -> Optional[RateLimitHit]:
        combined = (result.get("stderr") or "") + "\n" + (result.get("stdout") or "")
        for pat in _CLAUDE_PATTERNS:
            m = pat.search(combined)
            if m:
                ra = _RETRY_AFTER.search(combined)
                return RateLimitHit(
                    retry_after_s=float(ra.group(1)) if ra else None,
                    detail=m.group(0),
                )
        return None


_OPENAI_PATTERNS = (
    re.compile(r"rate_limit_exceeded", re.I),
    re.compile(r"\b429\b"),
)


class CodexRateLimitDetector:
    def classify(self, result: dict) -> Optional[RateLimitHit]:
        combined = (result.get("stderr") or "") + "\n" + (result.get("stdout") or "")
        for pat in _OPENAI_PATTERNS:
            m = pat.search(combined)
            if m:
                ra = _RETRY_AFTER.search(combined)
                return RateLimitHit(
                    retry_after_s=float(ra.group(1)) if ra else None,
                    detail=m.group(0),
                )
        return None


# Google's Gemini API uses RESOURCE_EXHAUSTED (gRPC-ish) alongside 429.
_GEMINI_PATTERNS = (
    re.compile(r"RESOURCE_EXHAUSTED"),
    re.compile(r"\b429\b"),
)


class GeminiRateLimitDetector:
    def classify(self, result: dict) -> Optional[RateLimitHit]:
        combined = (result.get("stderr") or "") + "\n" + (result.get("stdout") or "")
        for pat in _GEMINI_PATTERNS:
            m = pat.search(combined)
            if m:
                ra = _RETRY_AFTER.search(combined)
                return RateLimitHit(
                    retry_after_s=float(ra.group(1)) if ra else None,
                    detail=m.group(0),
                )
        return None


# ---------- Retry loop ---------------------------------------------------------


def with_rate_limit_retry(
    call: Callable[[], dict],
    detector: RateLimitDetector,
    policy: RateLimitPolicy,
    on_retry: Optional[Callable[[RateLimitEvent], None]],
    interface: str,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[float, float], float] = random.uniform,
) -> dict:
    """Invoke ``call`` until it succeeds or rate-limit attempts are exhausted.

    ``call`` must return the per-interface subprocess dict (``success``,
    ``stdout``, ``stderr``, ``returncode``, optionally ``token_usage``).
    Non-rate-limit failures are returned as-is for the caller to surface
    — they don't consume retry budget.
    """
    last: Optional[dict] = None
    result: dict = {}
    for attempt in range(1, policy.max_retries + 2):
        result = call()
        if result.get("success"):
            return result
        hit = detector.classify(result)
        if hit is None:
            return result
        if attempt > policy.max_retries:
            raise RateLimitExhausted(interface, attempt - 1, hit.detail)
        if hit.retry_after_s is not None:
            wait = min(hit.retry_after_s, policy.max_seconds)
        else:
            wait = min(policy.base_seconds * (2 ** (attempt - 1)), policy.max_seconds)
        wait += rand(0.0, policy.jitter_seconds)
        if on_retry is not None:
            on_retry(
                RateLimitEvent(
                    interface=interface,
                    attempt=attempt,
                    wait_seconds=wait,
                    detail=hit.detail,
                )
            )
        sleep(wait)
        last = result
    return last if last is not None else result
