"""Writer-preferring readers-writer lock.

Many threads may hold the read side concurrently; the write side is exclusive.
Waiting writers block new readers so a steady stream of searches cannot
starve an ingest. Not reentrant - never nest acquisitions of the same lock.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class RWLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer_active or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_active = True
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()
