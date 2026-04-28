"""Cancellable subprocess runner shared by all three CLI interfaces.

Slice Bug 1 (2026-04-22) — the worker's old cancel flow only checked
``run.status == CANCEL_REQUESTED`` between instances. That meant the
currently-running ``claude -p`` / ``codex`` / ``gemini`` CLI subprocess
kept running to completion (often 30+ minutes) before the shard ever
noticed the cancel. Workers that got SIGKILLed mid-instance left
zombie RQ worker records that tied up queue slots.

This helper replaces the plain ``subprocess.run`` call with a
``Popen`` in its own process group plus a cancel-polling wait loop.
On cancel, SIGTERM the group, wait a short grace period, then SIGKILL
if anything is still alive. Returns the same ``(rc, stdout, stderr)``
shape the interfaces already expect, with an extra ``cancelled`` flag.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import IO, Callable, List, Optional

# ``contextlib.suppress`` alias used as a short-name noise swallower for
# "drain whatever bytes exist, skip if the stream is weird". Avoids
# polluting the module namespace twice.
contextlib_suppress = contextlib.suppress


@dataclass(frozen=True)
class CancellableResult:
    """Mirrors the subset of ``subprocess.CompletedProcess`` the
    interfaces consume, plus ``timed_out`` and ``cancelled`` flags."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False


_POLL_INTERVAL_S = 1.0
_DEFAULT_GRACE_S = 3.0


def run_with_cancel(
    cmd: list[str],
    *,
    input: str,
    cwd: str,
    env: Optional[dict] = None,
    timeout_s: float,
    is_cancelled: Optional[Callable[[], bool]] = None,
    grace_s: float = _DEFAULT_GRACE_S,
) -> CancellableResult:
    """Drop-in for ``subprocess.run(capture_output=True, text=True)`` that
    honors an ``is_cancelled`` predicate.

    Semantics:
      * No ``is_cancelled`` (or always-False) → behaves like the old
        ``subprocess.run`` — single ``wait`` with a hard timeout.
      * ``is_cancelled()`` returns True at any point → SIGTERM the
        process group, wait up to ``grace_s``, SIGKILL if the tree is
        still alive. Returns ``CancellableResult(cancelled=True)`` with
        whatever bytes the process wrote before the signal.

    Requires POSIX semantics (``os.setsid`` → process group). The
    project is Mac/Linux only; no Windows support planned.
    """
    # ``start_new_session=True`` puts the child in its own process
    # group so ``os.killpg`` can reach the CLI *and* any grandchildren
    # it spawned (Claude Code in particular fans out). Without this,
    # SIGTERM on the direct child only kills the parent; grandchildren
    # keep the GPU / API calls running.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        text=True,
        start_new_session=True,
    )

    try:
        # Feed stdin + close — ``communicate(timeout=…)`` is the usual
        # path but we need control of the wait loop so we can poll
        # ``is_cancelled``. Writing the full input at once is fine: our
        # prompts are small (single-digit KB) and the CLI reads stdin
        # eagerly. For a cancellable communicate we'd need threads to
        # drain stdout/stderr without deadlocking on full pipe buffers,
        # so we fall back to ``communicate(timeout=…)`` once we know we
        # haven't been asked to cancel.
        stdout, stderr = _wait_cancellable(
            proc,
            input_text=input,
            timeout_s=timeout_s,
            is_cancelled=is_cancelled,
            grace_s=grace_s,
        )
    except subprocess.TimeoutExpired:
        # Cancel-path's ``_wait_cancellable`` already killed the group +
        # joined the drain threads + stashed partial stdout/stderr on
        # the proc before raising. Reading from ``proc.stdout`` here
        # would race the drain threads (now joined) and yield empty —
        # use the stash instead. No-cancel path has no drain threads
        # in play, so it falls through to ``_drain_stdio`` for the
        # post-kill read.
        if getattr(proc, "_cl_partial_stdout", None) is not None:
            stdout = proc._cl_partial_stdout  # type: ignore[attr-defined]
            stderr = proc._cl_partial_stderr  # type: ignore[attr-defined]
        else:
            _kill_process_group(proc, sig=signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=grace_s)
            stdout, stderr = _drain_stdio(proc)
        return CancellableResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr or f"Command timed out after {timeout_s} seconds",
            timed_out=True,
        )

    # Convention: a cancelled wait returns None tuples + sets the proc
    # into an exited state via kill. The flag is on the proc object.
    if getattr(proc, "_cl_cancelled", False):
        return CancellableResult(
            returncode=proc.returncode if proc.returncode is not None else -signal.SIGTERM,
            stdout=stdout or "",
            stderr=(stderr or "") + "\n[cl-benchmark] subprocess cancelled by worker\n",
            cancelled=True,
        )

    return CancellableResult(
        returncode=proc.returncode if proc.returncode is not None else 0,
        stdout=stdout or "",
        stderr=stderr or "",
    )


def _wait_cancellable(
    proc: subprocess.Popen,
    *,
    input_text: str,
    timeout_s: float,
    is_cancelled: Optional[Callable[[], bool]],
    grace_s: float,
) -> tuple[str, str]:
    """Split ``proc.communicate`` into a write + cancellable wait.

    For the no-cancel path we delegate straight to ``communicate`` so
    the standard deadlock-avoidance path runs unchanged. For the
    cancel path we write stdin up front + close, then poll
    ``is_cancelled`` with short waits until the process exits (safe
    for our CLI prompts since they're <10 KB — well under any pipe
    buffer).
    """
    # No cancel predicate → preserve the pre-Bug-1 single-call path.
    # Lets ``communicate`` manage stdin write + stdout drain in its
    # usual way so large outputs don't deadlock on full pipe buffers.
    if is_cancelled is None:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout_s)
        return stdout or "", stderr or ""

    # Cancel path: push stdin up front, close it, then poll ``wait``
    # instead of ``communicate`` — ``communicate`` tries to re-flush
    # stdin and errors on our already-closed pipe.
    #
    # Bug 2 (2026-04-28): we MUST drain stdout + stderr concurrently
    # with the wait loop. macOS pipe buffers are 64 KiB; ``proc.wait``
    # alone doesn't read from the pipes, so a child writing more than
    # 64 KiB to stdout (e.g. ``claude --output-format stream-json`` for
    # any non-trivial run) blocks on its next write, never reaches the
    # final ``result`` event, and eventually exits with its stdout
    # buffered at exactly 64 KiB. Draining the pipes in background
    # reader threads lets the child stream the full transcript through
    # — same mechanism ``proc.communicate`` uses internally for the
    # no-cancel path.
    if proc.stdin is not None:
        try:
            proc.stdin.write(input_text)
        finally:
            proc.stdin.close()

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    stdout_thread = (
        _start_drain_thread(proc.stdout, stdout_chunks) if proc.stdout is not None else None
    )
    stderr_thread = (
        _start_drain_thread(proc.stderr, stderr_chunks) if proc.stderr is not None else None
    )

    elapsed = 0.0
    while elapsed < timeout_s:
        if is_cancelled():
            _signal_cancel(proc, grace_s=grace_s)
            setattr(proc, "_cl_cancelled", True)
            return _join_drain_threads(stdout_thread, stderr_thread, stdout_chunks, stderr_chunks)
        try:
            proc.wait(timeout=_POLL_INTERVAL_S)
            return _join_drain_threads(stdout_thread, stderr_thread, stdout_chunks, stderr_chunks)
        except subprocess.TimeoutExpired:
            elapsed += _POLL_INTERVAL_S
            continue
    # Outer timeout — kill, join drain threads (they hit EOF on the
    # killed pipes), stash the partial bytes on the proc so the
    # caller's ``except TimeoutExpired`` block can surface them
    # without re-reading and racing the drain threads.
    _kill_process_group(proc, sig=signal.SIGKILL)
    with _suppress_timeout():
        proc.wait(timeout=grace_s)
    partial_out, partial_err = _join_drain_threads(
        stdout_thread, stderr_thread, stdout_chunks, stderr_chunks
    )
    setattr(proc, "_cl_partial_stdout", partial_out)
    setattr(proc, "_cl_partial_stderr", partial_err)
    raise subprocess.TimeoutExpired(proc.args, timeout_s)


def _drain_stdio(proc: subprocess.Popen) -> tuple[str, str]:
    """Read whatever stdout/stderr the process wrote before exiting.
    Called only after ``wait`` returned (or we killed the group), so
    the streams are EOF and won't block. Used by the timeout-expired
    branch in ``run_with_cancel`` — the cancel-path uses background
    drain threads instead (see ``_start_drain_thread``)."""
    stdout = ""
    stderr = ""
    if proc.stdout is not None:
        with contextlib_suppress(Exception):
            stdout = proc.stdout.read() or ""
    if proc.stderr is not None:
        with contextlib_suppress(Exception):
            stderr = proc.stderr.read() or ""
    return stdout, stderr


def _start_drain_thread(stream: IO[str], chunks: List[str]) -> threading.Thread:
    """Spawn a daemon thread that reads ``stream`` to EOF and appends
    each ``read()`` result to ``chunks``. Mirrors what
    ``subprocess.communicate`` does internally so the child doesn't
    block on a full pipe buffer (64 KiB on macOS) while we poll
    ``is_cancelled`` in the main thread."""

    def _drain() -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                chunks.append(chunk)
        except (OSError, ValueError):
            # Pipe closed mid-read (process killed, signal-induced
            # broken pipe). The wait/cancel path handles termination;
            # here we just stop draining and surface what we have.
            return

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    return t


def _join_drain_threads(
    stdout_thread: Optional[threading.Thread],
    stderr_thread: Optional[threading.Thread],
    stdout_chunks: List[str],
    stderr_chunks: List[str],
    *,
    join_timeout_s: float = 5.0,
) -> tuple[str, str]:
    """Wait for the drain threads to finish (process has exited / been
    killed → streams hit EOF → threads return), then assemble the
    accumulated chunks. Bounded join to avoid hanging on a stuck
    reader thread; callers tolerate empty/partial output on degraded
    paths."""
    if stdout_thread is not None:
        stdout_thread.join(timeout=join_timeout_s)
    if stderr_thread is not None:
        stderr_thread.join(timeout=join_timeout_s)
    return "".join(stdout_chunks), "".join(stderr_chunks)


def _signal_cancel(proc: subprocess.Popen, *, grace_s: float) -> None:
    """SIGTERM the group, wait ``grace_s``, SIGKILL any survivors."""
    _kill_process_group(proc, sig=signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc, sig=signal.SIGKILL)
        with _suppress_timeout():
            proc.wait(timeout=grace_s)


def _kill_process_group(proc: subprocess.Popen, *, sig: int) -> None:
    """``os.killpg`` the child's group, tolerating already-dead groups."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        # Already dead, or we raced with a natural exit. No-op.
        return


class _suppress_timeout:
    """Context manager that swallows ``TimeoutExpired`` so the
    cancel path can ``wait(timeout=…)`` without leaking the exception
    when the process refuses to die cleanly."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return isinstance(exc, subprocess.TimeoutExpired)
