from __future__ import annotations

from typing import Any

import pytest

from src.db import session_store


class _FakeConnection:
    """Records every executed query/params; returns canned rows for fetch."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        self.executed.append((query, params))
        return "OK"

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.executed.append((query, params))
        return self.rows


class _FakeAcquire:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _FakePool:
    """Mirrors asyncpg.Pool's ``acquire()`` context-manager shape."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.conn = _FakeConnection(rows)

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  hello   world \n there ", "hello world there"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_make_title(text: str, expected: str) -> None:
    assert session_store.make_title(text) == expected


def test_make_title_truncates_long_text() -> None:
    title = session_store.make_title("x" * 200)
    assert len(title) == 60
    assert title.endswith("…")


async def test_none_pool_is_a_noop() -> None:
    # None pool must never raise and must return empty/None equivalents.
    await session_store.touch_session(None, "s1", title_hint="hi", message_count=2)
    await session_store.delete_session(None, "s1")
    assert await session_store.list_sessions(None) == []


async def test_touch_session_executes_upsert_with_expected_params() -> None:
    pool = _FakePool()
    await session_store.touch_session(pool, "s1", title_hint="Hello world", message_count=3)

    assert len(pool.conn.executed) == 1
    query, params = pool.conn.executed[0]
    assert "INSERT INTO" in query
    assert "ON CONFLICT" in query
    assert params == ("s1", "Hello world", 3)


async def test_touch_session_swallows_execute_errors() -> None:
    class _FailingPool:
        def acquire(self) -> _FakeAcquire:
            raise RuntimeError("connection refused")

    # Best-effort: a broken pool must not propagate to the caller.
    await session_store.touch_session(_FailingPool(), "s1", title_hint="hi", message_count=1)


async def test_list_sessions_executes_select_and_maps_rows_to_dicts() -> None:
    row = {
        "session_id": "s1",
        "title": "hi",
        "created_at": None,
        "last_active": None,
        "message_count": 1,
    }
    pool = _FakePool(rows=[row])

    result = await session_store.list_sessions(pool)

    assert result == [row]
    query, params = pool.conn.executed[0]
    assert "SELECT" in query and "ORDER BY" in query
    assert params == ()


async def test_delete_session_executes_delete_with_session_id_param() -> None:
    pool = _FakePool()
    await session_store.delete_session(pool, "s1")

    query, params = pool.conn.executed[0]
    assert "DELETE FROM" in query
    assert params == ("s1",)
