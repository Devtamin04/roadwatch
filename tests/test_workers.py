import threading
import time

import numpy as np
import pytest

from roadwatch.perception.workers import LatestFrameWorker


def frame(seq):
    """Tiny image that encodes its seq, so fn can report which frame it saw."""
    return np.full((2, 2), seq, dtype=np.int64)


def seq_of(img):
    return int(img[0, 0])


@pytest.fixture
def workers():
    created = []
    yield created
    for w in created:
        w.stop()


def make(workers, fn, **kw):
    w = LatestFrameWorker(fn, **kw)
    workers.append(w)
    return w


def test_processes_only_latest_frame(workers):
    seen = []
    gate = threading.Event()

    def slow(img):
        gate.wait(2)
        seen.append(seq_of(img))
        return seq_of(img)

    w = make(workers, slow)
    w.submit(frame(0), "s", 0)
    time.sleep(0.05)  # worker is now blocked on frame 0
    for seq in range(1, 10):
        w.submit(frame(seq), "s", seq)
    gate.set()
    assert w.wait_idle()
    assert seen == [0, 9]  # frames 1..8 were overwritten
    assert w.stats.overwritten == 8
    assert w.get_latest("s", 9) == 9


def test_every_n_filters_frames(workers):
    seen = []
    w = make(workers, lambda img: seen.append(seq_of(img)), every_n=3)
    for seq in range(10):
        assert w.submit(frame(seq), "s", seq) == (seq % 3 == 0)
        w.wait_idle()
    assert seen == [0, 3, 6, 9]
    assert w.stats.skipped == 6


def test_old_session_result_never_returned(workers):
    gate = threading.Event()

    def slow(img):
        gate.wait(2)
        return "old"

    w = make(workers, slow)
    w.submit(frame(5), "A", 5)
    time.sleep(0.05)  # processing A/5
    w.reset("B")
    gate.set()  # A's result finishes after the reset
    assert w.wait_idle()
    assert w.get_latest("B", 5) is None
    assert w.get_latest("A", 5) is None  # A's result was discarded, not stored
    assert w.stats.discarded == 1


def test_new_session_on_submit_implicitly_resets(workers):
    w = make(workers, seq_of)
    w.submit(frame(3), "A", 3)
    w.wait_idle()
    assert w.get_latest("A", 3) == 3
    w.submit(frame(0), "B", 0)
    w.wait_idle()
    assert w.get_latest("A", 3) is None
    assert w.get_latest("B", 0) == 0


def test_stale_result_dropped(workers):
    w = make(workers, seq_of, max_staleness_frames=10)
    w.submit(frame(100), "s", 100)
    w.wait_idle()
    assert w.get_latest("s", 100) == 100
    assert w.get_latest("s", 110) == 100
    assert w.get_latest("s", 111) is None
    assert w.get_latest("s", 99) is None  # result from the "future": invalid


def test_fn_exception_does_not_kill_worker(workers):
    def flaky(img):
        if seq_of(img) == 1:
            raise RuntimeError("boom")
        return seq_of(img)

    w = make(workers, flaky)
    w.submit(frame(0), "s", 0)
    w.wait_idle()
    w.submit(frame(1), "s", 1)
    w.wait_idle()
    assert w.alive and w.stats.errors == 1
    assert w.get_latest("s", 1) == 0  # previous good result still valid
    w.submit(frame(2), "s", 2)
    w.wait_idle()
    assert w.get_latest("s", 2) == 2


def test_stop_is_clean(workers):
    w = make(workers, seq_of)
    w.submit(frame(0), "s", 0)
    w.stop(timeout=2)
    assert not w.alive
    assert w.submit(frame(1), "s", 1) is False
