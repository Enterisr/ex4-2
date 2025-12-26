from __future__ import annotations

import os
from typing import Iterable, Optional

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
    """Wrap an iterable with tqdm if available to show progress."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def resolve_workers(value: Optional[int]) -> int:
    """Resolve worker count, limiting to a sensible CPU bound when value is missing or zero."""
    if value is None or value == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return max(1, value)
