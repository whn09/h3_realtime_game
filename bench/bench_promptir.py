#!/usr/bin/env python3
"""P0 checklist item 8 -- is PromptIR's latency survivable?

The reference implementation measures 8.1s mean on 4 requests. That is the
single largest non-GPU item in the beat budget, and 4 samples is not a
measurement. This script produces the numbers the design actually needs:

  * prefill vs decode split, via time-to-first-token
  * p50 AND p95 (p95 is what decides whether users see a spinner)
  * cached vs uncached arms, to size what prompt caching really buys
  * output token count, split across the three IR sections, to size the
    "template sections 2 and 3 instead of generating them" lever

Prompt caching only skips prefill. If the split comes back decode-dominated --
which is the expectation -- then caching is a cost win and a small latency win,
and the real lever is output length. This script measures which is true.

    export AWS_REGION=us-west-2
    python bench_promptir.py --system-file ir_system_prompt.txt --reps 30

Dump the system prompt from the reference implementation first, e.g.:
    python -c "from h3ir.guides import *; ..."  > ir_system_prompt.txt
(check that module for its actual builder name; the point is to benchmark the
real prompt, since its size is exactly what determines the prefill cost)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from common import print_table, summarise, write_csv, write_json

# Bedrock model IDs carry an `anthropic.` prefix. Haiku 4.5 is what the
# reference implementation measured, so it is the comparable baseline.
DEFAULT_MODEL = "anthropic.claude-haiku-4-5"

# Stand-in beats, used when no --requests-file is given, so the script is
# runnable immediately. Shaped like real Director output: one action, one shot.
DEFAULT_REQUESTS = [
    {"id": "ruins_wide", "user": "固定镜头，大远景。黄昏，一名旅人独自站在废墟城市的天台边缘，"
                                "风吹动他的外套，他缓缓抬头望向远处倒塌的信号塔。时长15秒，16:9。"},
    {"id": "market_medium", "user": "手持跟拍，中景。雨夜的霓虹市场，女侦探快步穿过摊位之间，"
                                   "回头看了一眼身后的人群。时长15秒，16:9。"},
    {"id": "cultivation_cu", "user": "固定镜头，特写。青年修士闭眼盘坐在山巅，掌心一枚玉符缓缓亮起，"
                                    "他睁开眼睛。时长15秒，16:9。"},
    {"id": "corridor_pov", "user": "第一人称视角，缓慢推进。空无一人的医院走廊，应急灯闪烁，"
                                  "尽头的一扇门正在缓缓打开。时长15秒，16:9。"},
]


def load_requests(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_REQUESTS
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            rows.append({"id": obj.get("id", f"req{len(rows)}"), "user": obj["user"]})
    return rows


def split_ir_sections(text: str) -> list[str]:
    """The IR format is three sections separated by a single newline.

    Splitting lets us measure how much of the decode budget the two nearly
    static sections (soundscape, music) consume -- which is the size of the
    prize for templating them from the world bible instead of generating them.
    """
    parts = [p for p in text.split("\n") if p.strip()]
    return parts[:3] if len(parts) >= 3 else parts


def run_once(
    client: Any, model: str, system: str, user: str, max_tokens: int, cached: bool
) -> dict[str, Any]:
    """One streaming call, instrumented for TTFT.

    The cache breakpoint goes on the system block only. The per-beat user
    message stays *after* it, so the volatile part never invalidates the cached
    prefix -- the standard placement, and the one PromptIR wants.
    """
    system_param: Any
    if cached:
        system_param = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    else:
        system_param = [{"type": "text", "text": system}]

    start = time.perf_counter()
    ttft_ms: float | None = None
    chunks: list[str] = []

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_param,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                chunks.append(event.delta.text)
        final = stream.get_final_message()

    total_ms = (time.perf_counter() - start) * 1000.0
    text = "".join(chunks)
    usage = final.usage
    sections = split_ir_sections(text)

    row: dict[str, Any] = {
        "cached_arm": cached,
        "ttft_ms": round(ttft_ms or total_ms, 1),
        "decode_ms": round(total_ms - (ttft_ms or total_ms), 1),
        "total_ms": round(total_ms, 1),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "stop_reason": final.stop_reason,
        "n_sections": len(sections),
        "out_chars": len(text),
        "sec1_chars": len(sections[0]) if len(sections) > 0 else 0,
        "sec2_chars": len(sections[1]) if len(sections) > 1 else 0,
        "sec3_chars": len(sections[2]) if len(sections) > 2 else 0,
        "text": text,
    }
    # Output tokens are only reported in aggregate, so apportion them by
    # character share to estimate the per-section decode cost.
    if row["out_chars"]:
        tail_share = (row["sec2_chars"] + row["sec3_chars"]) / row["out_chars"]
        row["tail_section_share"] = round(tail_share, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-file", required=True,
                    help="the real PromptIR system prompt (guides + 36 rules + gold examples)")
    ap.add_argument("--requests-file", default=None, help="JSONL with {id,user} per line")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--reps", type=int, default=30,
                    help="calls per arm; 30+ before trusting a p95")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="ceiling only -- keep above the real IR length so nothing truncates")
    ap.add_argument("--arms", default="cached,uncached")
    ap.add_argument("--out-prefix", default="results/bench_promptir")
    args = ap.parse_args()

    system = Path(args.system_file).read_text(encoding="utf-8")
    requests = load_requests(args.requests_file)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    from anthropic import AnthropicBedrockMantle

    client = AnthropicBedrockMantle(aws_region=args.region)

    print(f"model={args.model} region={args.region}")
    print(f"system prompt: {len(system)} chars  requests: {len(requests)}  reps/arm: {args.reps}")

    rows: list[dict[str, Any]] = []

    for arm in arms:
        cached = arm == "cached"
        print(f"\n=== arm: {arm}")
        if cached:
            # The first cached call is a cache *write*, which is slower and
            # would poison the measurement. Burn one to warm it.
            print("    warming cache (this call writes it and is excluded)...")
            warm = run_once(client, args.model, system, requests[0]["user"], args.max_tokens, True)
            print(f"    write={warm['cache_write_tokens']}tok read={warm['cache_read_tokens']}tok "
                  f"total={warm['total_ms']:.0f}ms")

        for i in range(args.reps):
            req = requests[i % len(requests)]
            try:
                row = run_once(client, args.model, system, req["user"], args.max_tokens, cached)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i + 1}/{args.reps}] {req['id']}: FAILED {exc}")
                rows.append({"arm": arm, "request_id": req["id"], "rep": i, "error": str(exc)[:300]})
                continue
            row |= {"arm": arm, "request_id": req["id"], "rep": i}
            rows.append(row)
            print(f"  [{i + 1}/{args.reps}] {req['id']}: ttft={row['ttft_ms']:.0f}ms "
                  f"decode={row['decode_ms']:.0f}ms total={row['total_ms']:.0f}ms "
                  f"out={row['output_tokens']}tok cache_read={row['cache_read_tokens']}tok")

    ok = [r for r in rows if "error" not in r]
    # Keep the generated IR out of the CSV (it is long); it lives in the JSON.
    write_csv(Path(f"{args.out_prefix}.csv"), [{k: v for k, v in r.items() if k != "text"} for r in ok])
    write_json(Path(f"{args.out_prefix}_samples.json"), ok[:12])

    table = []
    for arm in arms:
        group = [r for r in ok if r["arm"] == arm]
        if not group:
            continue
        ttft = summarise([r["ttft_ms"] for r in group])
        decode = summarise([r["decode_ms"] for r in group])
        total = summarise([r["total_ms"] for r in group])
        table.append({
            "arm": arm, "n": total["n"],
            "ttft_p50": ttft["p50"], "ttft_p95": ttft["p95"],
            "decode_p50": decode["p50"], "decode_p95": decode["p95"],
            "total_p50": total["p50"], "total_p95": total["p95"],
            "out_tok_mean": round(statistics.fmean([r["output_tokens"] for r in group])),
            "cache_read_mean": round(statistics.fmean([r["cache_read_tokens"] for r in group])),
        })
    print_table(
        "PROMPTIR LATENCY (ms)", table,
        ["arm", "n", "ttft_p50", "ttft_p95", "decode_p50", "decode_p95",
         "total_p50", "total_p95", "out_tok_mean", "cache_read_mean"],
    )

    print("\nDECISIONS UNLOCKED")
    by_arm = {r["arm"]: r for r in table}

    cached_row, uncached_row = by_arm.get("cached"), by_arm.get("uncached")

    if cached_row and cached_row["cache_read_mean"] == 0:
        print("  !! cache_read_input_tokens is 0 across the cached arm -- caching is NOT "
              "engaging. Either the system prompt is below the model's minimum cacheable "
              "prefix, or something in the prefix varies between calls. Fix before trusting "
              "any cached numbers.")

    if cached_row and uncached_row:
        ttft_saved = uncached_row["ttft_p50"] - cached_row["ttft_p50"]
        total_saved = uncached_row["total_p50"] - cached_row["total_p50"]
        print(f"  caching saves {ttft_saved:.0f}ms of TTFT and {total_saved:.0f}ms of total (p50)")
        if cached_row["total_p50"]:
            print(f"  -> that is {total_saved / uncached_row['total_p50'] * 100:.0f}% of total "
                  "latency; the rest is decode and caching cannot touch it")

    ref = cached_row or uncached_row
    if ref:
        split_pct = ref["ttft_p50"] / ref["total_p50"] * 100 if ref["total_p50"] else 0
        print(f"  prefill is {split_pct:.0f}% of p50 latency, decode is {100 - split_pct:.0f}%")
        if split_pct < 35:
            print("     -> decode-dominated, as expected. Output length is the lever, not "
                  "caching. Proceed with templating the soundscape/music sections.")
        else:
            print("     -> prefill is a bigger share than expected; shrinking the system "
                  "prompt (fewer gold examples) is also worth measuring.")

        tails = [r.get("tail_section_share") for r in ok if r.get("tail_section_share") is not None]
        if tails:
            share = statistics.fmean(tails)
            saved_ms = ref["decode_p50"] * share
            print(f"  IR sections 2+3 (soundscape, music) are {share * 100:.0f}% of output chars")
            print(f"     -> templating them should cut roughly {saved_ms:.0f}ms of decode, "
                  f"taking p50 to about {ref['total_p50'] - saved_ms:.0f}ms")

        print("\n  BUDGET CHECK (director 3.0s + promptIR + generation, vs the 19.3s window)")
        for label, promptir_ms in (
            ("measured p50", ref["total_p50"]),
            ("measured p95", ref["total_p95"]),
            ("p95 after templating sections 2+3",
             ref["total_p95"] - ref["decode_p95"] * (statistics.fmean(tails) if tails else 0)),
            ("local distilled 7B target", 1000.0),
        ):
            for gen_s in (9.0, 11.0):
                chain = 3.0 + promptir_ms / 1000.0 + gen_s
                verdict = "FITS" if chain <= 19.3 else f"OVER by {chain - 19.3:.1f}s"
                print(f"    {label:<38} + gen {gen_s:.0f}s = {chain:>5.1f}s -> {verdict}")

    truncated = [r for r in ok if r.get("stop_reason") == "max_tokens"]
    if truncated:
        print(f"\n  !! {len(truncated)} responses hit max_tokens and were truncated -- raise "
              "--max-tokens; truncated IR would fail the validator in production")

    print(f"\nrows -> {args.out_prefix}.csv   sample IR -> {args.out_prefix}_samples.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
