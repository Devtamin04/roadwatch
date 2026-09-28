import os
import threading

import pytest

from roadwatch.cpu import call_in_thread, hybrid_core_sets, placement

linux_only = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")


def _thread_state():
    tid = threading.get_native_id()
    return os.sched_getaffinity(tid), os.getpriority(os.PRIO_PROCESS, tid)


@linux_only
def test_call_in_thread_applies_placement_and_nice_to_new_threads():
    cpus = {min(os.sched_getaffinity(0))}
    base_nice = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())

    def make_child_and_report():
        box = {}
        t = threading.Thread(target=lambda: box.update(state=_thread_state()))
        t.start()
        t.join()
        return _thread_state(), box["state"]

    own, child = call_in_thread(make_child_and_report, nice=5, cpus=cpus)
    assert own == child == (cpus, max(base_nice, 5))  # inherited by threads it creates
    assert _thread_state()[0] != cpus or len(os.sched_getaffinity(0)) == 1  # caller untouched


def test_call_in_thread_propagates_errors_and_values():
    assert call_in_thread(lambda: 42) == 42
    with pytest.raises(KeyError):
        call_in_thread(lambda: {}["x"], nice=1)


def test_placement_disabled_or_non_hybrid():
    assert placement(False) == (None, None)
    fg, bg = placement(True)
    if hybrid_core_sets() is None:
        assert (fg, bg) == (None, None)
    else:
        assert fg and bg and not (fg & bg)
