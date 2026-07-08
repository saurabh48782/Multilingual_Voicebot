"""Per-session serialization so concurrent requests for the same session_id
can't race the checkpointer.

session_id is client-supplied, unauthenticated string with no server-side
turn-taking; a double-submit (or two tabs sharing localStorage) would let two
`graph.ainvoke` calls interleave against the same thread_id and corrupt the
checkpointed history.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class SessionLocks:
    def __init__(self) -> None:
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get(self, session_id: str) -> asyncio.Lock:
        return self._locks[session_id]


__all__ = ["SessionLocks"]
