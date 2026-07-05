from __future__ import annotations

from src.db import session_store


def test_make_title_collapses_whitespace() -> None:
    assert session_store.make_title("  hello   world \n there ") == "hello world there"


def test_make_title_truncates_long_text() -> None:
    title = session_store.make_title("x" * 200)
    assert len(title) == 60
    assert title.endswith("…")


def test_make_title_handles_empty() -> None:
    assert session_store.make_title("") == ""
    assert session_store.make_title("   ") == ""


async def test_none_pool_is_a_noop() -> None:
    # None pool must never raise and must return empty/None equivalents.
    await session_store.touch_session(None, "s1", title_hint="hi", message_count=2)
    await session_store.delete_session(None, "s1")
    assert await session_store.list_sessions(None) == []
