"""Tests for the rate-limit retry helpers (Phase 4)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.retry import (
    ClaudeRateLimitDetector,
    CodexRateLimitDetector,
    GeminiRateLimitDetector,
    RateLimitExhausted,
    RateLimitPolicy,
    with_rate_limit_retry,
)


# ---------- Detector fidelity --------------------------------------------------


def test_claude_detector_matches_rate_limit_error():
    det = ClaudeRateLimitDetector()
    hit = det.classify({"stderr": "Error: rate_limit_error (rpm exceeded)"})
    assert hit is not None
    assert hit.detail == "rate_limit_error"
    assert hit.retry_after_s is None


def test_claude_detector_matches_overloaded_error_with_retry_after():
    det = ClaudeRateLimitDetector()
    hit = det.classify(
        {"stderr": "overloaded_error\nRetry-After: 45"}
    )
    assert hit is not None
    assert hit.retry_after_s == 45.0


def test_claude_detector_ignores_unrelated_failure():
    det = ClaudeRateLimitDetector()
    assert det.classify({"stderr": "ModuleNotFoundError: datasets"}) is None


def test_codex_detector_matches_rate_limit_exceeded():
    det = CodexRateLimitDetector()
    assert det.classify({"stderr": "rate_limit_exceeded (1 rpm)"}) is not None


def test_codex_detector_matches_bare_429():
    det = CodexRateLimitDetector()
    hit = det.classify({"stderr": "HTTP 429 Too Many Requests"})
    assert hit is not None


def test_gemini_detector_matches_resource_exhausted():
    det = GeminiRateLimitDetector()
    hit = det.classify(
        {"stderr": "RESOURCE_EXHAUSTED\nretry-after: 12"}
    )
    assert hit is not None
    assert hit.retry_after_s == 12.0


# ---------- Retry loop ---------------------------------------------------------


def _fake_rl_result(extra: str = "") -> dict:
    return {"success": False, "stderr": f"rate_limit_error {extra}", "stdout": "", "returncode": 1}


def _fake_success() -> dict:
    return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}


class _SleepRecorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def test_retry_succeeds_after_two_hits():
    sleeps = _SleepRecorder()
    events: list = []
    calls = iter([_fake_rl_result(), _fake_rl_result(), _fake_success()])

    out = with_rate_limit_retry(
        call=lambda: next(calls),
        detector=ClaudeRateLimitDetector(),
        policy=RateLimitPolicy(
            max_retries=5, base_seconds=1.0, max_seconds=10.0, jitter_seconds=0.0
        ),
        on_retry=events.append,
        interface="claude",
        sleep=sleeps,
        rand=lambda lo, hi: 0.0,
    )
    assert out["success"] is True
    assert len(events) == 2
    assert [e.attempt for e in events] == [1, 2]
    # Exponential backoff: base * 2^(attempt-1) = 1, 2
    assert sleeps.sleeps == [1.0, 2.0]


def test_retry_honors_parsed_retry_after():
    sleeps = _SleepRecorder()
    calls = iter(
        [
            {"success": False, "stderr": "rate_limit_error\nRetry-After: 7"},
            _fake_success(),
        ]
    )
    with_rate_limit_retry(
        call=lambda: next(calls),
        detector=ClaudeRateLimitDetector(),
        policy=RateLimitPolicy(max_retries=5, base_seconds=1.0, jitter_seconds=0.0),
        on_retry=None,
        interface="claude",
        sleep=sleeps,
        rand=lambda lo, hi: 0.0,
    )
    assert sleeps.sleeps == [7.0]


def test_retry_caps_at_max_seconds():
    sleeps = _SleepRecorder()
    calls = iter(
        [
            {"success": False, "stderr": "rate_limit_error\nRetry-After: 999"},
            _fake_success(),
        ]
    )
    with_rate_limit_retry(
        call=lambda: next(calls),
        detector=ClaudeRateLimitDetector(),
        policy=RateLimitPolicy(
            max_retries=5, base_seconds=1.0, max_seconds=30.0, jitter_seconds=0.0
        ),
        on_retry=None,
        interface="claude",
        sleep=sleeps,
        rand=lambda lo, hi: 0.0,
    )
    assert sleeps.sleeps == [30.0]  # Retry-After=999 clamped to max_seconds=30


def test_retry_raises_on_exhaustion():
    sleeps = _SleepRecorder()
    # 6 consecutive hits — max_retries=5 means the 6th raises.
    calls = iter([_fake_rl_result() for _ in range(6)])

    with pytest.raises(RateLimitExhausted) as ei:
        with_rate_limit_retry(
            call=lambda: next(calls),
            detector=ClaudeRateLimitDetector(),
            policy=RateLimitPolicy(
                max_retries=5, base_seconds=1.0, max_seconds=10.0, jitter_seconds=0.0
            ),
            on_retry=None,
            interface="claude",
            sleep=sleeps,
            rand=lambda lo, hi: 0.0,
        )
    assert ei.value.attempts == 5
    assert ei.value.interface == "claude"
    # 5 sleeps happen (between attempts 1-6); the 6th hit raises before sleep.
    assert len(sleeps.sleeps) == 5


def test_retry_surfaces_non_rate_limit_failure():
    """Non-rate-limit failures (e.g., missing dataset) return as-is without
    consuming retry budget — the caller surfaces them unchanged."""
    result = with_rate_limit_retry(
        call=lambda: {"success": False, "stderr": "FileNotFoundError: foo.json"},
        detector=ClaudeRateLimitDetector(),
        policy=RateLimitPolicy(max_retries=5),
        on_retry=None,
        interface="claude",
        sleep=lambda s: None,
        rand=lambda lo, hi: 0.0,
    )
    assert result["success"] is False
    assert "FileNotFoundError" in result["stderr"]


def test_retry_short_circuits_on_immediate_success():
    result = with_rate_limit_retry(
        call=lambda: _fake_success(),
        detector=ClaudeRateLimitDetector(),
        policy=RateLimitPolicy(max_retries=5),
        on_retry=None,
        interface="claude",
        sleep=lambda s: None,
        rand=lambda lo, hi: 0.0,
    )
    assert result["success"] is True
