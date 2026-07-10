from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api.routers.documents import safe_destination


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scheme.pdf", "scheme.pdf"),  # plain filename lands in upload dir
        ("../../etc/passwd.txt", "passwd.txt"),  # traversal components stripped
    ],
)
def test_accepted_filenames(name: str, expected: str, tmp_path: Path) -> None:
    dest = safe_destination(name, upload_dir=tmp_path)
    assert dest.parent == tmp_path.resolve()
    assert dest.name == expected


def test_reupload_same_name_overwrites_same_path(tmp_path: Path) -> None:
    a = safe_destination("doc.txt", upload_dir=tmp_path)
    b = safe_destination("doc.txt", upload_dir=tmp_path)
    assert a == b  # doc_id stability: re-ingest replaces, never duplicates


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "..",
        ".hidden.pdf",
        "a\\..\\b.pdf",
        "evil.exe",
        "noextension",
        " padded.pdf",
    ],
)
def test_rejected_filenames(name: str | None, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        safe_destination(name, upload_dir=tmp_path)
    assert exc.value.status_code == 422
