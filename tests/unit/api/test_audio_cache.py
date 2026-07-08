import os
import time
from pathlib import Path

from src.api.audio_cache import AudioCache


def test_put_and_path_for_roundtrip(tmp_path: Path) -> None:
    cache = AudioCache(cache_dir=tmp_path, ttl_seconds=3600, max_files=10)
    audio_id = cache.put(b"WAVBYTES")
    path = cache.path_for(audio_id)
    assert path is not None
    assert path.read_bytes() == b"WAVBYTES"


def test_expired_files_swept_on_put(tmp_path: Path) -> None:
    cache = AudioCache(cache_dir=tmp_path, ttl_seconds=3600, max_files=10)
    old_id = cache.put(b"OLD")
    old_path = cache.path_for(old_id)
    assert old_path is not None
    stale = time.time() - 7200
    os.utime(old_path, (stale, stale))

    cache.put(b"NEW")
    assert cache.path_for(old_id) is None


def test_file_count_cap_evicts_oldest(tmp_path: Path) -> None:
    cache = AudioCache(cache_dir=tmp_path, ttl_seconds=3600, max_files=3)
    ids = []
    for i in range(6):
        ids.append(cache.put(f"AUDIO{i}".encode()))
        # Distinct mtimes so eviction order is deterministic.
        path = cache.path_for(ids[-1])
        assert path is not None
        ts = time.time() - (6 - i)
        os.utime(path, (ts, ts))

    cache._sweep()
    remaining = [audio_id for audio_id in ids if cache.path_for(audio_id) is not None]
    assert len(remaining) <= 3
    # Newest entries survive
    assert all(i in remaining for i in ids[-3:])
