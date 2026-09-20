#!/usr/bin/env python3
"""P0 checklist items 3, 4, 5, 10 -- the latency matrix.

  #3  How much slower is i2va (first-frame conditioned) than t2va? Every
      gameplay beat except the opening is i2va, so if it is materially slower
      the 1.5s gap in DESIGN.md section 2.3 gets worse, not better.
  #4  The quality/latency curve over `num_inference_steps` and `quality`. This
      is the primary degradation knob for the fallback ladder, so we need the
      actual shape of it, not a guess.
  #5  Is one instance serving 2 concurrent requests as good as 2 instances
      serving 1 each? This decides whether "2 SGLang instances" really means
      "2 GPU slots", which the whole scheduling design rests on.
  #10 Latency cost of 480 vs 768 short edge -- i.e. whether there is headroom
      to ship at a higher resolution.

Latency is measured here; *quality* still has to be judged by eye, so every
cell's output URLs are printed grouped for side-by-side viewing.

    python bench_h3.py --wrapper http://127.0.0.1:8000 --reps 3
    python bench_h3.py --wrapper http://gpu1:8000 --wrapper http://gpu2:8000 \
        --concurrency-test
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from common import make_test_keyframe, print_table, summarise, write_csv, write_json

PROMPT = (
    "Fixed camera, medium shot. A lone figure stands on a windswept ridge at dusk, "
    "coat snapping in the wind, slowly turning to look back over one shoulder.\n"
    "Wind across open ground, fabric snapping, distant birds.\n"
    "Sparse low strings and a single sustained cello note, 60 BPM, minor key."
)


@dataclass(frozen=True)
class Cell:
    task: str
    steps: int
    quality: str
    short_edge: int
    seconds: float

    @property
    def label(self) -> str:
        return f"{self.task}/{self.steps}st/{self.quality}/{self.short_edge}p/{self.seconds:g}s"


async def run_one(
    client: httpx.AsyncClient, wrapper: str, cell: Cell, keyframe: str | None, rep: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": PROMPT,
        "task": "fl2va" if cell.task == "i2va" else cell.task,
        "seconds": cell.seconds,
        "short_edge": cell.short_edge,
        "quality": cell.quality,
        "num_inference_steps": cell.steps,
        # Fixed seed so quality differences across cells are attributable to the
        # parameter under test rather than to sampling noise.
        "seed": 1234,
        # Post-processing stays ON: these stages are part of the real budget and
        # the point of the exercise is the real budget.
        "archive": False,
    }
    if cell.task == "i2va":
        body["first_frame"] = keyframe

    row: dict[str, Any] = {
        "cell": cell.label, "task": cell.task, "steps": cell.steps,
        "quality": cell.quality, "short_edge": cell.short_edge,
        "seconds": cell.seconds, "rep": rep, "wrapper": wrapper,
    }

    try:
        resp = await client.post(f"{wrapper.rstrip('/')}/generate", json=body)
    except Exception as exc:  # noqa: BLE001
        row |= {"ok": False, "error": str(exc)[:300]}
        return row

    if resp.status_code != 200:
        row |= {"ok": False, "error": f"http {resp.status_code}: {resp.text[:300]}"}
        return row

    data = resp.json()
    t = data["timings"]
    media = data["media"]
    row |= {
        "ok": True,
        "total_ms": round(t["total_ms"], 1),
        "sglang_ms": round(t["sglang_ms"], 1),
        "download_ms": round(t["download_ms"], 1),
        "last_frame_ms": round(t["last_frame_ms"], 1),
        "faststart_ms": round(t["faststart_ms"], 1),
        "probe_ms": round(t["probe_ms"], 1),
        "post_ms": round(
            t["download_ms"] + t["probe_ms"] + t["last_frame_ms"]
            + t["poster_ms"] + t["faststart_ms"] + t["analyze_ms"] + t["publish_ms"], 1
        ),
        "width": media.get("width"),
        "height": media.get("height"),
        "actual_duration_ms": media.get("duration_ms"),
        "fps": media.get("fps"),
        "has_audio": media.get("has_audio"),
        "size_bytes": media.get("size_bytes"),
        "video_url": data.get("video_url"),
    }
    return row


async def run_matrix(args: argparse.Namespace, keyframe: str) -> list[dict[str, Any]]:
    cells = [
        Cell(task, steps, quality, edge, seconds)
        for task, steps, quality, edge, seconds in itertools.product(
            args.tasks, args.steps, args.quality, args.short_edge, args.seconds
        )
    ]
    rows: list[dict[str, Any]] = []
    wrapper = args.wrapper[0]

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        total = len(cells) * args.reps
        done = 0
        for cell in cells:
            for rep in range(args.reps):
                # Strictly serial: any overlap would contaminate the very
                # latency numbers we are trying to measure.
                row = await run_one(client, wrapper, cell, keyframe, rep)
                rows.append(row)
                done += 1
                status = (
                    f"{row['sglang_ms']:.0f}ms sglang, {row['post_ms']:.0f}ms post, "
                    f"audio={row['has_audio']}"
                    if row.get("ok") else f"FAILED {row.get('error')}"
                )
                print(f"[{done}/{total}] {cell.label} rep{rep}: {status}")
    return rows


async def run_concurrency_test(args: argparse.Namespace, keyframe: str) -> dict[str, Any]:
    """Checklist #5: does one instance batch usefully, or do we need two?

    Mode A -- 2 requests in flight on ONE wrapper. Requires that wrapper to run
              with MAX_CONCURRENT=2, otherwise its semaphore serialises them and
              the result is meaningless (we detect and warn).
    Mode B -- 1 request on each of TWO wrappers, in parallel. This is the
              topology DESIGN.md assumes.

    If A's wall clock is close to B's, one instance is worth two slots and the
    whole scheduling story gets cheaper. If A is ~2x B, "2 instances = 2 slots"
    is confirmed and single-instance batching is off the table.
    """
    cell = Cell("i2va", args.steps[0], args.quality[0], args.short_edge[0], args.seconds[0])
    out: dict[str, Any] = {"cell": cell.label}

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        health = await client.get(f"{args.wrapper[0].rstrip('/')}/healthz")
        slots = health.json().get("slots_total", 1)
        out["wrapper_a_slots"] = slots
        if slots < 2:
            print("WARNING: wrapper A reports MAX_CONCURRENT=1, so mode A will be "
                  "serialised by the wrapper, not by the GPU. Restart it with "
                  "MAX_CONCURRENT=2 for a valid measurement.")

        # Baseline: one request alone.
        solo = await run_one(client, args.wrapper[0], cell, keyframe, 0)
        out["solo_sglang_ms"] = solo.get("sglang_ms")
        print(f"solo: {solo.get('sglang_ms')}ms")

        # Mode A: two in flight on one instance.
        start = asyncio.get_running_loop().time()
        a_rows = await asyncio.gather(*[
            run_one(client, args.wrapper[0], cell, keyframe, i) for i in range(2)
        ])
        out["mode_a_wall_ms"] = round((asyncio.get_running_loop().time() - start) * 1000, 1)
        out["mode_a_per_request_ms"] = [r.get("sglang_ms") for r in a_rows]
        print(f"mode A (2 on one instance): wall={out['mode_a_wall_ms']}ms "
              f"per-request={out['mode_a_per_request_ms']}")

        # Mode B: one request on each of two instances.
        if len(args.wrapper) >= 2:
            start = asyncio.get_running_loop().time()
            b_rows = await asyncio.gather(*[
                run_one(client, w, cell, keyframe, i)
                for i, w in enumerate(args.wrapper[:2])
            ])
            out["mode_b_wall_ms"] = round((asyncio.get_running_loop().time() - start) * 1000, 1)
            out["mode_b_per_request_ms"] = [r.get("sglang_ms") for r in b_rows]
            print(f"mode B (1 on each of two): wall={out['mode_b_wall_ms']}ms "
                  f"per-request={out['mode_b_per_request_ms']}")
        else:
            print("only one --wrapper given; skipping mode B")

    solo_ms = out.get("solo_sglang_ms")
    if solo_ms and out.get("mode_a_wall_ms"):
        ratio = out["mode_a_wall_ms"] / solo_ms
        out["mode_a_slowdown_vs_solo"] = round(ratio, 2)
        print(f"\n  mode A wall clock is {ratio:.2f}x a solo request")
        if ratio < 1.35:
            print("  -> one instance absorbs 2 concurrent requests cheaply: a single "
                  "instance may be worth 2 slots, which halves the GPU needed per session")
        elif ratio > 1.8:
            print("  -> requests serialise on the GPU: '2 instances = 2 slots' confirmed, "
                  "both branches must go to different instances (as designed)")
        else:
            print("  -> partial overlap; treat as ~1.5 slots and keep branches on "
                  "separate instances for predictable latency")
    return out


def analyse(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok = [r for r in rows if r.get("ok")]
    failed = [r for r in rows if not r.get("ok")]

    cells: dict[str, list[dict[str, Any]]] = {}
    for row in ok:
        cells.setdefault(row["cell"], []).append(row)

    table = []
    for label, group in cells.items():
        sg = summarise([r["sglang_ms"] for r in group])
        post = summarise([r["post_ms"] for r in group])
        total = summarise([r["total_ms"] for r in group])
        table.append({
            "cell": label,
            "n": sg["n"],
            "sglang_p50": sg["p50"],
            "sglang_p95": sg["p95"],
            "post_p50": post["p50"],
            "total_p50": total["p50"],
            "total_p95": total["p95"],
            "audio": all(r["has_audio"] for r in group),
            "res": f"{group[0]['width']}x{group[0]['height']}",
        })
    table.sort(key=lambda r: r["total_p50"])
    print_table(
        "LATENCY BY CELL (ms)", table,
        ["cell", "n", "sglang_p50", "sglang_p95", "post_p50", "total_p50", "total_p95",
         "audio", "res"],
    )

    if failed:
        print(f"\n{len(failed)} FAILED runs:")
        for row in failed[:10]:
            print(f"  {row['cell']} rep{row['rep']}: {row.get('error')}")

    print("\nDECISIONS UNLOCKED")

    # #3 i2va vs t2va
    def p50_for(pred) -> float | None:
        vals = [r["sglang_ms"] for r in ok if pred(r)]
        return statistics.median(vals) if vals else None

    for edge in args.short_edge:
        for steps in args.steps:
            t2va = p50_for(lambda r: r["task"] == "t2va" and r["short_edge"] == edge and r["steps"] == steps)
            i2va = p50_for(lambda r: r["task"] == "i2va" and r["short_edge"] == edge and r["steps"] == steps)
            if t2va and i2va:
                delta = i2va - t2va
                print(f"  #3 {edge}p/{steps}st: i2va {i2va:.0f}ms vs t2va {t2va:.0f}ms "
                      f"({delta:+.0f}ms, {delta / t2va * 100:+.0f}%)")
                if delta > 1500:
                    print("     -> i2va overhead eats the remaining budget; plan on "
                          "depth-2 speculative prefetch (needs 2 more GPU slots)")

    # #4 steps curve
    if len(args.steps) > 1:
        print("  #4 steps curve (i2va, p50 sglang_ms):")
        for steps in sorted(args.steps):
            val = p50_for(lambda r: r["task"] == "i2va" and r["steps"] == steps)
            if val:
                print(f"     {steps:>3} steps: {val:>7.0f}ms   -- judge quality by eye below")

    # #10 resolution
    if len(args.short_edge) > 1:
        print("  #10 resolution cost (i2va, p50 sglang_ms):")
        for edge in sorted(args.short_edge):
            val = p50_for(lambda r: r["task"] == "i2va" and r["short_edge"] == edge)
            if val:
                print(f"     short_edge {edge:>4}: {val:>7.0f}ms")

    # Budget verdict: the number this whole script exists to produce.
    best = [r for r in ok if r["task"] == "i2va"]
    if best:
        gen_p95 = summarise([r["total_ms"] for r in best])["p95"] / 1000.0
        print(f"\nBUDGET CHECK (using i2va p95 total = {gen_p95:.1f}s)")
        for promptir in (8.1, 6.5, 5.0, 1.0):
            chain = 3.0 + promptir + gen_p95
            verdict = "FITS" if chain <= 19.3 else f"OVER by {chain - 19.3:.1f}s"
            print(f"  director 3.0s + promptIR {promptir:>4.1f}s + gen {gen_p95:.1f}s "
                  f"= {chain:>5.1f}s vs 19.3s window -> {verdict}")

    print("\nQUALITY REVIEW -- open these side by side and judge by eye:")
    for label, group in sorted(cells.items()):
        if group[0].get("video_url"):
            print(f"  {label}: {group[0]['video_url']}")


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", action="append", default=[],
                    help="h3-wrapper base URL; repeat for the concurrency test")
    ap.add_argument("--tasks", default="t2va,i2va",
                    help="comma list of t2va,i2va (i2va = fl2va with first frame only)")
    ap.add_argument("--steps", default="32,50")
    ap.add_argument("--quality", default="high")
    ap.add_argument("--short-edge", default="480")
    ap.add_argument("--seconds", default="15")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--keyframe", default=None)
    ap.add_argument("--concurrency-test", action="store_true")
    ap.add_argument("--out-prefix", default="results/bench_h3")
    args = ap.parse_args()

    if not args.wrapper:
        args.wrapper = ["http://127.0.0.1:8000"]
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.steps = [int(v) for v in args.steps.split(",")]
    args.quality = [q.strip() for q in args.quality.split(",")]
    args.short_edge = [int(v) for v in args.short_edge.split(",")]
    args.seconds = [float(v) for v in args.seconds.split(",")]

    keyframe_path = (
        Path(args.keyframe) if args.keyframe
        else make_test_keyframe(Path("/tmp/h3game-bench/keyframe.png"))
    )
    keyframe = str(keyframe_path.resolve())

    if args.concurrency_test:
        result = await run_concurrency_test(args, keyframe)
        write_json(Path(f"{args.out_prefix}_concurrency.json"), result)
        return 0

    rows = await run_matrix(args, keyframe)
    n = write_csv(Path(f"{args.out_prefix}.csv"), rows)
    analyse(rows, args)
    print(f"\n{n} rows -> {args.out_prefix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
