"""Structured stdout progress for long-running training and evaluation loops."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def format_progress(tag: str, **fields: Any) -> str:
    """Render one progress line: ``[tag] key=value ...``."""
    parts = [f"[{tag}]"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


@dataclass
class ProgressTracker:
    """Emit start/progress/end banners with elapsed time, rate, and ETA."""

    tag: str
    total: int
    phase: str = "default"
    enabled: bool = True
    _start: float = field(default_factory=time.monotonic)
    _done: int = 0
    _last_log: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._log_every = max(1, self.total // 10) if self.total else 1

    def banner(self, **extra: Any) -> None:
        if not self.enabled:
            return
        print(
            format_progress(
                self.tag,
                phase=extra.pop("phase", self.phase),
                **extra,
            ),
            flush=True,
        )

    def tick(self, count: int = 1, *, force: bool = False, **extra: Any) -> None:
        if not self.enabled:
            return
        self._done += count
        should_log = force or self._done >= self.total or self._done % self._log_every == 0
        if not should_log:
            return
        elapsed = time.monotonic() - self._start
        rate = self._done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self._done)
        eta = remaining / rate if rate > 0 else 0.0
        print(
            format_progress(
                self.tag,
                phase=self.phase,
                step=f"{self._done}/{self.total}",
                elapsed=f"{elapsed:.0f}s",
                rate=f"{rate:.2f}/s",
                eta=f"{eta:.0f}s",
                **extra,
            ),
            flush=True,
        )
        self._last_log = time.monotonic()

    def end(self, **extra: Any) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self._start
        rate = self._done / elapsed if elapsed > 0 else 0.0
        if "elapsed" not in extra:
            extra["elapsed"] = f"{elapsed:.0f}s"
        if "rate" not in extra:
            extra["rate"] = f"{rate:.2f}/s"
        print(
            format_progress(
                self.tag,
                phase=self.phase,
                step=f"{self._done}/{self.total}",
                **extra,
            ),
            flush=True,
        )
