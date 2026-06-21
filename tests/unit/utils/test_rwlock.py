"""Unit tests for the readers-writer lock guarding the RAG indexes."""

import threading
import time
from collections.abc import Callable

from src.utils.rwlock import RWLock


def _start(target: Callable[[], None]) -> threading.Thread:
    t = threading.Thread(target=target)
    t.start()
    return t


def _join_all(*threads: threading.Thread, timeout: float = 5.0) -> None:
    for t in threads:
        t.join(timeout=timeout)


def test_readers_run_concurrently() -> None:
    lock = RWLock()
    barrier = threading.Barrier(2, timeout=2)
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            with lock.read():
                barrier.wait()  # only passes if both readers are inside at once
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [_start(reader) for _ in range(2)]
    _join_all(*threads)
    assert not errors, f"readers did not overlap: {errors}"


def test_writer_excludes_readers() -> None:
    lock = RWLock()
    order: list[str] = []
    write_acquired = threading.Event()
    release_write = threading.Event()

    def writer() -> None:
        with lock.write():
            write_acquired.set()
            release_write.wait(timeout=2)
            order.append("write")

    def reader() -> None:
        assert write_acquired.wait(timeout=2)
        with lock.read():
            order.append("read")

    tw = _start(writer)
    tr = _start(reader)

    assert write_acquired.wait(timeout=2)
    release_write.set()
    _join_all(tw, tr)
    assert order == ["write", "read"]


def test_waiting_writer_blocks_new_readers() -> None:
    """Writer preference: a queued writer goes before readers that arrive later."""
    lock = RWLock()
    order: list[str] = []
    reader1_in = threading.Event()
    release_reader1 = threading.Event()

    def reader1() -> None:
        with lock.read():
            reader1_in.set()
            release_reader1.wait(timeout=2)

    def writer() -> None:
        with lock.write():
            order.append("write")

    def reader2() -> None:
        with lock.read():
            order.append("read2")

    t1 = _start(reader1)
    assert reader1_in.wait(timeout=2)

    tw = _start(writer)
    time.sleep(0.1)  # let the writer queue before the late reader arrives
    t2 = _start(reader2)
    time.sleep(0.1)

    release_reader1.set()
    _join_all(t1, tw, t2)
    assert order == ["write", "read2"]
