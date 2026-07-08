"""Disk-backed TTS audio cache.

TTS output bytes are written to `{AUDIO_CACHE_DIR}/{audio_id}.{ext}` keyed by a
random UUID. Files older than `AUDIO_TTL_SECONDS` are swept on each `put()`
call (cheap, no background task needed for a single-process app).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from src.utils.config import audio_cache_dir, cfg


class AudioCache:
    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_seconds: int | None = None,
        max_files: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir or audio_cache_dir
        self.ttl_seconds = ttl_seconds or cfg["audio"]["cache_ttl_seconds"]
        self.max_files = max_files or int(cfg["audio"].get("cache_max_files", 1000))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def put(self, audio: bytes, content_type: str = "audio/wav") -> str:
        """Persist audio. Returns the audio_id (without extension)."""
        self._sweep()
        audio_id = uuid.uuid4().hex
        extension = "mp3" if content_type == "audio/mpeg" else "wav"
        path = self.cache_dir / f"{audio_id}.{extension}"
        path.write_bytes(audio)
        return audio_id

    def path_for(self, audio_id: str) -> Path | None:
        """Resolve to the on-disk path. Returns None if not found."""
        for ext in ("wav", "mp3"):
            candidate = self.cache_dir / f"{audio_id}.{ext}"
            if candidate.is_file():
                return candidate
        return None

    def _sweep(self) -> int:
        """Evict expired files (older than ttl_seconds), then trim to max_files
        by evicting oldest-first if still over the cap. Returns count removed."""
        cutoff = time.time() - self.ttl_seconds
        removed = 0
        live: list[tuple[float, Path]] = []
        for child in self.cache_dir.iterdir():
            if not child.is_file():
                continue
            try:
                mtime = child.stat().st_mtime
                if mtime < cutoff:
                    child.unlink()
                    removed += 1
                else:
                    live.append((mtime, child))
            except OSError:
                continue

        # Hard cap on file count: a request burst within the TTL window must
        # not grow the cache unboundedly. Evict oldest first.
        overflow = len(live) - self.max_files
        if overflow > 0:
            live.sort()
            for _, child in live[:overflow]:
                try:
                    child.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed
