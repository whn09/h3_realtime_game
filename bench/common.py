"""Shared helpers for the P0 benchmark suite.

P0's whole purpose is to decide, before any UI is written, whether the 19.3s
window in DESIGN.md can actually absorb the pipeline. That decision hinges on
p95, not means -- a p50 that fits and a p95 that doesn't means users see
spinners. So everything here reports percentiles.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value at or above pct of the sample.

    Nearest-rank rather than interpolated, because an interpolated p95 on 30
    samples invents a number between two real observations. Here p95 is always
    an actually-observed latency, which is what a budget decision should rest on.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[max(1, min(len(ordered), rank)) - 1]


def summarise(values: Sequence[float]) -> dict[str, float]:
    clean = [v for v in values if v == v]  # drop NaN
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean": round(statistics.fmean(clean), 1),
        "p50": round(percentile(clean, 50), 1),
        "p95": round(percentile(clean, 95), 1),
        "min": round(min(clean), 1),
        "max": round(max(clean), 1),
    }


@contextmanager
def timed() -> Any:
    """Yields a one-element list that receives elapsed ms on exit."""
    holder: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = (time.perf_counter() - start) * 1000.0


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    # Union of keys, preserving first-seen order, so partial failures still
    # produce a readable file.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_test_keyframe(dest: Path, width: int = 854, height: int = 480) -> Path:
    """Synthesise a plausible keyframe so probing does not depend on any asset.

    Deliberately not a flat colour: a gradient plus a high-contrast subject
    gives the model something to continue from, and gives the drift metrics
    non-degenerate sharpness/saturation baselines.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    # Broadcast both axes to the full grid up front; channels that depend only
    # on ys would otherwise stay (height, 1) and fail to stack.
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )

    # Dusk sky over dark ground -- reads as an establishing shot.
    r = 0.15 + 0.60 * (1.0 - ys) ** 2 + 0.10 * xs
    g = 0.12 + 0.35 * (1.0 - ys) ** 2
    b = 0.20 + 0.45 * (1.0 - ys)
    arr = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    arr[int(height * 0.72):, :, :] *= 0.25  # ground plane

    img = Image.fromarray((arr * 255).astype("uint8"), "RGB")
    draw = ImageDraw.Draw(img)
    # A silhouetted figure: gives the sharpness metric real high-frequency edges.
    cx, base = int(width * 0.42), int(height * 0.78)
    draw.ellipse([cx - 9, base - 74, cx + 9, base - 56], fill=(8, 8, 10))
    draw.polygon(
        [(cx - 14, base), (cx - 10, base - 56), (cx + 10, base - 56), (cx + 14, base)],
        fill=(8, 8, 10),
    )
    draw.ellipse(
        [int(width * 0.74), int(height * 0.16), int(width * 0.86), int(height * 0.30)],
        fill=(255, 238, 200),
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return dest


def print_table(title: str, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no rows)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
