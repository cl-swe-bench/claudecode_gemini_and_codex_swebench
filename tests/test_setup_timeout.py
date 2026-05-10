"""Setup-phase wall-clock timeout in ``process_instance``.

Spec: cl-benchmark/docs/stuck-sample-recovery-spec.md (Fix #3).
Pre-fix the harness's setup phase (git clone + MCP injection +
prompt formatting) had no timeout — only the post-spawn CLI wait
inside ``cancellable_subprocess`` did. A hung git clone could pin
a shard indefinitely, bypassing the existing timeout and
silently stranding the run.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_swe_agent import (  # noqa: E402
    SETUP_TIMEOUT_S,
    SetupTimeoutError,
    _setup_timeout,
)


def test_setup_timeout_disabled_when_zero():
    """``SETUP_TIMEOUT_S=0`` (or ``<=0``) yields without installing a
    handler — the context manager is a no-op. Used by tests + as an
    emergency rollback."""
    # A long-running operation inside the context must complete
    # uninterrupted.
    with _setup_timeout(0):
        time.sleep(0.05)  # ~50ms, harmless


def test_setup_timeout_fires_on_blocking_work():
    """A blocking operation longer than the budget raises
    ``SetupTimeoutError`` via SIGALRM."""
    with pytest.raises(SetupTimeoutError):
        with _setup_timeout(1):
            # Sleep > 1s — the SIGALRM handler raises during the call.
            time.sleep(2)


def test_setup_timeout_cancels_alarm_on_normal_exit():
    """After a successful pass through, ``signal.alarm(0)`` cancels
    the pending alarm — a subsequent slow op outside the context
    must not get hit."""
    with _setup_timeout(1):
        pass  # finishes immediately
    # Sleep longer than the budget; no SetupTimeoutError should fire
    # because the alarm was cancelled.
    time.sleep(1.2)


def test_setup_timeout_module_constant_is_configured():
    """The default budget is non-trivial (covers any reasonable
    SWE-bench Pro clone) but bounded; the env-var override surfaces
    via ``SETUP_TIMEOUT_S``. Pin the default so a refactor doesn't
    silently lower it to a tight value that bites large repos."""
    assert SETUP_TIMEOUT_S >= 300  # at least 5 min for big repos
