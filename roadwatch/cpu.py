"""CPU placement helpers (Linux): hybrid P/E-core detection, affinity, priority.

Affinity and nice values are per thread on Linux and are inherited by threads
(and processes) created afterwards. Inference sessions and decoders are
therefore constructed inside a thread that already has the wanted placement,
so their internal thread pools inherit it.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)


def hybrid_core_sets() -> tuple[set[int], set[int]] | None:
    """(performance, efficiency) logical CPU ids on a hybrid Intel CPU, else None."""

    def read(path: str) -> set[int]:
        cpus: set[int] = set()
        with open(path, encoding="utf-8") as f:
            for part in f.read().strip().split(","):
                a, _, b = part.partition("-")
                cpus.update(range(int(a), int(b or a) + 1))
        return cpus

    try:
        pcores = read("/sys/devices/cpu_core/cpus")
        ecores = read("/sys/devices/cpu_atom/cpus")
    except (OSError, ValueError):
        return None
    allowed = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else pcores | ecores
    pcores, ecores = pcores & allowed, ecores & allowed
    return (pcores, ecores) if pcores and ecores else None


def pin_current_thread(cpus: set[int] | None) -> None:
    """Restrict the calling thread (and threads it creates later) to `cpus`."""
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(threading.get_native_id(), cpus)
    except OSError as e:
        log.warning("could not set CPU affinity: %s", e)


def lower_thread_priority(nice: int) -> None:
    """Raise the calling thread's nice value (lower priority). Linux only; no-op elsewhere.

    Threads created afterwards by this thread (e.g. an inference library's thread
    pool) inherit the value.
    """
    if nice <= 0 or not hasattr(os, "setpriority"):
        return
    try:
        tid = threading.get_native_id()
        os.setpriority(os.PRIO_PROCESS, tid, max(os.getpriority(os.PRIO_PROCESS, tid), nice))
    except OSError as e:  # e.g. restricted container
        log.warning("could not lower thread priority: %s", e)


def call_in_thread(fn: Callable[[], Any], nice: int = 0,
                   cpus: set[int] | None = None) -> Any:
    """Run fn in a fresh thread with lowered priority / restricted CPUs; return its result.

    Used to construct inference sessions so their internal thread pools start
    (and stay) at that priority and on those CPUs.
    """
    if nice <= 0 and not cpus:
        return fn()
    box: dict[str, Any] = {}

    def target() -> None:
        pin_current_thread(cpus)
        lower_thread_priority(nice)
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    t = threading.Thread(target=target, name="rw-init")
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def placement(enabled: bool = True) -> tuple[set[int] | None, set[int] | None]:
    """(foreground, background) CPU sets: P-cores / E-cores on a hybrid CPU, else (None, None)."""
    if not enabled:
        return None, None
    cores = hybrid_core_sets()
    return cores if cores else (None, None)
