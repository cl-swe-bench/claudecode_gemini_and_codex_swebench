"""Unit tests for ``utils.cancellable_subprocess.run_with_cancel``.

Bug 1 fix — when the worker flips ``is_cancelled`` True mid-run, the
helper SIGTERMs the CLI's process group (grace period, then SIGKILL)
and surfaces a ``cancelled=True`` flag back up. These tests use a
tiny ``sleep`` shell invocation as a stand-in for a long-running CLI.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.cancellable_subprocess import run_with_cancel  # noqa: E402


def test_happy_path_no_cancel():
    """No cancel predicate → behaves like plain ``subprocess.run``."""
    result = run_with_cancel(
        ["bash", "-c", "echo hello; exit 0"],
        input="",
        cwd=os.getcwd(),
        env=None,
        timeout_s=5,
        is_cancelled=None,
    )
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert not result.cancelled
    assert not result.timed_out


def test_cancel_fires_midflight_and_reports_cancelled():
    """Predicate flips True mid-sleep → process group is SIGTERMed
    within ~1s and the result carries ``cancelled=True``."""
    cancel_at = time.monotonic() + 0.5  # flip 500ms in

    def is_cancelled() -> bool:
        return time.monotonic() >= cancel_at

    start = time.monotonic()
    # A 30-second sleep that we expect to get killed at ~0.5s.
    result = run_with_cancel(
        ["bash", "-c", "sleep 30"],
        input="",
        cwd=os.getcwd(),
        env=None,
        timeout_s=60,
        is_cancelled=is_cancelled,
        grace_s=2.0,
    )
    elapsed = time.monotonic() - start

    assert result.cancelled, f"expected cancelled=True, got {result}"
    # Should exit within ~2s of the cancel flipping (1s poll + 1s grace).
    # The whole thing should take well under 10s even on a slow box.
    assert elapsed < 10.0, f"cancel took {elapsed}s; SIGTERM path is not working"
    assert "cancelled by worker" in result.stderr


def test_timeout_still_fires_when_cancel_never_flips():
    """Predicate that never fires + short timeout → timed_out=True
    without the cancel flag. Preserves pre-Bug-1 timeout semantics."""
    result = run_with_cancel(
        ["bash", "-c", "sleep 30"],
        input="",
        cwd=os.getcwd(),
        env=None,
        timeout_s=2,
        is_cancelled=lambda: False,
        grace_s=1.0,
    )
    assert result.timed_out
    assert not result.cancelled


def test_cancel_reaches_grandchildren_via_process_group():
    """The whole point of ``start_new_session=True`` — SIGTERM reaches
    grandchildren, not just the direct child. Spawn bash → sleep and
    verify the sleep dies too."""
    # bash -c invokes sleep as a child; without process-group signalling
    # the bash would die but sleep would keep running.
    cancel_at = time.monotonic() + 0.3

    def is_cancelled() -> bool:
        return time.monotonic() >= cancel_at

    start = time.monotonic()
    result = run_with_cancel(
        # ``exec`` would make bash replace itself with sleep — use the
        # non-exec form to get a real grandchild.
        ["bash", "-c", "sleep 30 & wait"],
        input="",
        cwd=os.getcwd(),
        env=None,
        timeout_s=60,
        is_cancelled=is_cancelled,
        grace_s=2.0,
    )
    elapsed = time.monotonic() - start
    assert result.cancelled
    # Same ~2s ceiling — if grandchild lived on, the ``wait`` would
    # keep bash alive and we'd hit the 60s timeout instead.
    assert elapsed < 10.0


@pytest.mark.parametrize("returncode", [0, 1, 2])
def test_non_cancel_exit_codes_propagate(returncode):
    """Fast-exiting commands that don't trigger cancel return their
    real return code with ``cancelled=False``."""
    result = run_with_cancel(
        ["bash", "-c", f"exit {returncode}"],
        input="",
        cwd=os.getcwd(),
        env=None,
        timeout_s=5,
        is_cancelled=lambda: False,
    )
    assert result.returncode == returncode
    assert not result.cancelled
    assert not result.timed_out


def test_cancel_path_streams_more_than_pipe_buffer():
    """Bug 2 (2026-04-28) regression guard. Cancel-path callers (worker
    runs are always cancel-path) must capture stdout > 64 KiB — the
    macOS pipe-buffer cap. Without concurrent draining, ``proc.wait``
    blocks the child once the buffer fills, the run completes with
    exactly 65 536 bytes, and the final ``result`` event (with token
    counts + cost) is lost. We emit ~256 KiB of stdout to verify the
    drain-thread fix.
    """

    # 256 KiB of "x\n" lines + a sentinel last line. If we see the
    # sentinel, draining worked: the child wasn't blocked at 64 KiB.
    payload_lines = 16384
    script = (
        "for i in $(seq 1 {n}); do printf 'xxxxxxxxxxxxxxxx\\n'; done; "
        "echo '__SENTINEL_LAST_LINE__'"
    ).format(n=payload_lines)

    result = run_with_cancel(
        ["bash", "-c", script],
        input="",
        cwd=os.getcwd(),
        env=None,
        timeout_s=15,
        is_cancelled=lambda: False,
    )
    assert result.returncode == 0
    assert not result.cancelled
    assert not result.timed_out
    # Total output is well above 64 KiB; the sentinel is the very last
    # line and only renders if the parent drained the pipe throughout.
    assert len(result.stdout) > 200 * 1024
    assert "__SENTINEL_LAST_LINE__" in result.stdout
