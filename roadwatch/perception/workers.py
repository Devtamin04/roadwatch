"""Background worker that always processes the most recent frame.

Slow perception modules (signs, lanes) run here so the main loop never waits
for them. The input slot holds a single frame: a newer submission overwrites
an older one that has not started processing yet. Results carry
(session_id, seq) so the caller can reject stale or cross-session results.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    session_id: str
    seq: int
    value: Any
    latency_ms: float


@dataclass
class WorkerStats:
    submitted: int = 0  # accepted into the slot
    skipped: int = 0  # rejected by every_n
    overwritten: int = 0  # dropped because a newer frame replaced it
    processed: int = 0
    errors: int = 0
    discarded: int = 0  # finished after a reset; result thrown away


class LatestFrameWorker:
    def __init__(
        self,
        fn: Callable[[np.ndarray], Any],
        every_n: int = 1,
        max_staleness_frames: int = 10,
        name: str = "worker",
    ):
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        self.fn = fn
        self.every_n = every_n
        self.max_staleness_frames = max_staleness_frames
        self.name = name
        self.stats = WorkerStats()

        self._cond = threading.Condition()
        self._slot: tuple[str, int, np.ndarray] | None = None
        self._result: WorkerResult | None = None
        self._session: str | None = None
        self._busy = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"rw-{name}", daemon=True)
        self._thread.start()

    # -- producer side ---------------------------------------------------
    def submit(self, image: np.ndarray, session_id: str, seq: int) -> bool:
        """Offer a frame. Returns True if accepted (seq % every_n == 0).

        A frame from a different session implicitly resets the worker.
        """
        if seq % self.every_n != 0:
            self.stats.skipped += 1
            return False
        with self._cond:
            if not self._running:
                return False
            if session_id != self._session:
                self._reset_locked(session_id)
            if self._slot is not None:
                self.stats.overwritten += 1
            self._slot = (session_id, seq, image)
            self.stats.submitted += 1
            self._cond.notify()
        return True

    def get_latest_result(self, session_id: str, current_seq: int) -> WorkerResult | None:
        """Latest result if it belongs to session_id and is not too old."""
        with self._cond:
            r = self._result
        if r is None or r.session_id != session_id:
            return None
        age = current_seq - r.seq
        if age < 0 or age > self.max_staleness_frames:
            return None
        return r

    def get_latest(self, session_id: str, current_seq: int) -> Any | None:
        r = self.get_latest_result(session_id, current_seq)
        return None if r is None else r.value

    def reset(self, session_id: str) -> None:
        """Start a new session (new video / seek): drop pending frame and old result."""
        with self._cond:
            self._reset_locked(session_id)

    def _reset_locked(self, session_id: str) -> None:
        self._session = session_id
        if self._slot is not None:
            self.stats.overwritten += 1
        self._slot = None
        self._result = None

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until no frame is pending or being processed (for tests/benchmarks)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._slot is not None or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def stop(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._running = False
            self._slot = None
            self._cond.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("%s worker did not stop within %.1fs", self.name, timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    # -- worker thread ---------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and self._slot is None:
                    self._cond.wait()
                if not self._running:
                    return
                session_id, seq, image = self._slot
                self._slot = None
                self._busy = True

            t0 = time.perf_counter()
            try:
                value = self.fn(image)
                ok = True
            except Exception:  # noqa: BLE001 - a module failure must not kill the pipeline
                log.exception("%s worker: fn failed on seq %d", self.name, seq)
                ok = False
            latency_ms = (time.perf_counter() - t0) * 1000

            with self._cond:
                self._busy = False
                if not ok:
                    self.stats.errors += 1
                elif session_id != self._session:
                    self.stats.discarded += 1
                else:
                    self.stats.processed += 1
                    if self._result is None or seq >= self._result.seq:
                        self._result = WorkerResult(session_id, seq, value, latency_ms)
                self._cond.notify_all()
