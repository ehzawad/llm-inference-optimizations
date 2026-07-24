#!/usr/bin/env python3
"""Aggregate a multi-GPU matrix run into comparative tables.

Consumes a PARENT directory whose subdirectories are individual placement x
shape runs produced by gpu_matrix.sh (``<config>-<shape>/`` each with
benchmark-*.json, experiment.json and config.json). Every subdir is re-validated
with the same fail-closed rules as report.py: a subdir with a failed/incomplete
point keeps its data visible but marks the config FAILED and the whole matrix
exits non-zero.

Emitted tables (per shape):
  * placement x concurrency aggregate throughput (tok/s)
  * prefill/decode split: TTFT p50/p95 and steady-state decode tok/s
  * DP vs TP vs single-card, and 2-GPU-aggregate vs sum-of-single-cards
  * A5000 vs A6000 single-card ratio

Usage: matrix_report.py <parent_dir> [--expect matrix.json]
                        [--out-md NAME] [--out-json NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report  # noqa: E402  (sibling module; reuse the fail-closed validators)


SINGLE_CARD = {"vllm-a5000", "vllm-a6000", "llamacpp-a5000", "llamacpp-a6000"}


def fmt(x: Any, nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "-"


def ratio(a: Any, b: Any) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return f"{a / b:.2f}x"
    return "-"


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "tps": row.get("output_tokens_per_s"),
        "tpm": row.get("output_tokens_per_min"),
        "ttft_p50": report.g(row, "ttft_s", "p50"),
        "ttft_p95": report.g(row, "ttft_s", "p95"),
        "decode_p50": report.g(row, "decode_tokens_per_s", "p50"),
        "decode_p95": report.g(row, "decode_tokens_per_s", "p95"),
        "lat_p50": report.g(row, "latency_s", "p50"),
        "lat_p95": report.g(row, "latency_s", "p95"),
        "vram_peak": report.g(row, "telemetry", "mem_used_mib", "peak"),
        "gpu_util": report.g(row, "telemetry", "gpu_util_pct", "median"),
    }


def load_config_dir(subdir: Path) -> dict[str, Any]:
    """Validate one placement x shape run; return its per-concurrency metrics."""
    info: dict[str, Any] = {
        "dir": subdir.name,
        "config": None,
        "shape": None,
        "engine": None,
        "precision": None,
        "ok": False,
        "reasons": [],
        "points": {},   # concurrency -> metrics (+ per-row ok)
    }

    config_path = subdir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            info.update({
                "config": cfg.get("config"),
                "shape": cfg.get("shape"),
                "engine": cfg.get("engine"),
                "precision": cfg.get("precision"),
            })
        except (OSError, json.JSONDecodeError) as exc:
            info["reasons"].append(f"unreadable config.json: {exc}")

    # Infer config/shape from the directory name when config.json is absent.
    if info["config"] is None or info["shape"] is None:
        name = subdir.name
        for shape in ("balanced", "prefill", "decode"):
            if name.endswith(f"-{shape}"):
                info["config"] = info["config"] or name[: -len(shape) - 1]
                info["shape"] = info["shape"] or shape
                break

    try:
        rows = report.load_rows(subdir)
        spec = report.expected_run_spec(subdir, require=True)
    except report.ReportError as exc:
        info["reasons"].append(str(exc))
        return info

    observed_points: set[int] = set()
    observed_tags: set[str] = set()
    all_ok = True
    for row in rows:
        ok, reasons = report.validate_row(row, spec)
        c = row.get("concurrency")
        if isinstance(c, int) and not isinstance(c, bool):
            observed_points.add(c)
            metrics = row_metrics(row)
            metrics["ok"] = ok
            info["points"][c] = metrics
        tag = row.get("tag")
        if isinstance(tag, str) and tag:
            observed_tags.add(tag)
        if not ok:
            all_ok = False
            info["reasons"].append(f"{row.get('_source_file')}: " + "; ".join(reasons))

    missing_points = sorted(spec.points - observed_points)
    missing_tags = sorted(spec.tags - observed_tags)
    if missing_points:
        info["reasons"].append("missing concurrency points: " + str(missing_points))
    if missing_tags:
        info["reasons"].append("missing tags: " + str(missing_tags))
    info["ok"] = all_ok and not missing_points and not missing_tags
    return info


def get(configs: dict[str, dict], config: str, c: int, field: str) -> Any:
    entry = configs.get(config)
    if not entry:
        return None
    point = entry["points"].get(c)
    return point.get(field) if point else None


def build_markdown(shapes: dict[str, dict[str, dict]], overall_ok: bool) -> str:
    lines: list[str] = []
    lines.append("# Multi-GPU comparative benchmark matrix\n")
    lines.append(f"Matrix status: **{'PASS' if overall_ok else 'FAILED / PARTIAL'}**.\n")
    lines.append(
        "Within-engine placement comparisons are primary; cross-engine numbers "
        "(llama.cpp Q6_K vs vLLM bf16) differ in precision and are secondary.\n"
    )

    for shape in sorted(shapes):
        configs = shapes[shape]
        all_c = sorted({c for cfg in configs.values() for c in cfg["points"]})
        present = sorted(configs)
        lines.append(f"\n## Shape: `{shape}`\n")
        failed = [name for name, cfg in configs.items() if not cfg["ok"]]
        if failed:
            lines.append(
                "FAILED / incomplete config(s): "
                + ", ".join(f"`{name}`" for name in sorted(failed))
                + " (data shown is diagnostic only).\n"
            )

        # 1. Aggregate throughput: placement x concurrency.
        lines.append("### Aggregate output throughput (tok/s) — placement × concurrency\n")
        lines.append("| placement | " + " | ".join(f"C={c}" for c in all_c) + " |")
        lines.append("|" + "---|" * (len(all_c) + 1))
        for name in present:
            cells = [fmt(get(configs, name, c, "tps")) for c in all_c]
            flag = "" if configs[name]["ok"] else " ⚠"
            lines.append(f"| {name}{flag} | " + " | ".join(cells) + " |")

        # 2. Prefill / decode split.
        lines.append("\n### Prefill (TTFT) vs decode split\n")
        lines.append(
            "| placement | C | TTFT p50 (s) | TTFT p95 (s) | decode tok/s p50 "
            "| decode tok/s p95 | latency p50 (s) | latency p95 (s) | VRAM peak MiB | GPU util med % |"
        )
        lines.append("|" + "---|" * 10)
        for name in present:
            for c in all_c:
                if c not in configs[name]["points"]:
                    continue
                m = configs[name]["points"][c]
                lines.append(
                    f"| {name} | {c} | {fmt(m['ttft_p50'], 3)} | {fmt(m['ttft_p95'], 3)} "
                    f"| {fmt(m['decode_p50'])} | {fmt(m['decode_p95'])} "
                    f"| {fmt(m['lat_p50'], 2)} | {fmt(m['lat_p95'], 2)} "
                    f"| {fmt(m['vram_peak'], 0)} | {fmt(m['gpu_util'], 0)} |"
                )

        # 3. DP vs TP vs single-card, and 2-GPU vs sum-of-single-cards.
        has_two_gpu = ("vllm-dp2" in configs) or ("vllm-tp2" in configs)
        has_singles = ("vllm-a5000" in configs) and ("vllm-a6000" in configs)
        if has_two_gpu and has_singles:
            lines.append("\n### 2-GPU (vLLM) vs single-card throughput (tok/s)\n")
            lines.append(
                "Note: dp2 offers total concurrency C split ~C/2 per card, so its "
                "load-matched comparator is the single cards at C/2. Both framings shown.\n"
            )
            lines.append(
                "| C | a5000 | a6000 | sum@C (a5000+a6000) | tp2 | dp2 "
                "| dp2/best-single | tp2/best-single | dp2@C / sum@(C/2) |"
            )
            lines.append("|" + "---|" * 9)
            for c in all_c:
                a5 = get(configs, "vllm-a5000", c, "tps")
                a6 = get(configs, "vllm-a6000", c, "tps")
                tp2 = get(configs, "vllm-tp2", c, "tps")
                dp2 = get(configs, "vllm-dp2", c, "tps")
                summ = (a5 + a6) if isinstance(a5, (int, float)) and isinstance(a6, (int, float)) else None
                best_single = None
                if isinstance(a5, (int, float)) or isinstance(a6, (int, float)):
                    best_single = max(v for v in (a5, a6) if isinstance(v, (int, float)))
                half = c // 2
                a5h = get(configs, "vllm-a5000", half, "tps")
                a6h = get(configs, "vllm-a6000", half, "tps")
                sum_half = (a5h + a6h) if isinstance(a5h, (int, float)) and isinstance(a6h, (int, float)) else None
                lines.append(
                    f"| {c} | {fmt(a5)} | {fmt(a6)} | {fmt(summ)} | {fmt(tp2)} | {fmt(dp2)} "
                    f"| {ratio(dp2, best_single)} | {ratio(tp2, best_single)} "
                    f"| {ratio(dp2, sum_half) if half in configs.get('vllm-a5000', {}).get('points', {}) else '-'} |"
                )

        # 4. A5000 vs A6000 single-card ratio (per engine when both present).
        for lo, hi, label in (
            ("vllm-a5000", "vllm-a6000", "vLLM bf16"),
            ("llamacpp-a5000", "llamacpp-a6000", "llama.cpp Q6_K"),
        ):
            if lo in configs and hi in configs:
                lines.append(f"\n### A5000 vs A6000 single-card ({label})\n")
                lines.append("| C | A5000 tok/s | A6000 tok/s | A6000/A5000 |")
                lines.append("|---|---|---|---|")
                for c in all_c:
                    a = get(configs, lo, c, "tps")
                    b = get(configs, hi, c, "tps")
                    if a is None and b is None:
                        continue
                    lines.append(f"| {c} | {fmt(a)} | {fmt(b)} | {ratio(b, a)} |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--expect", type=Path, default=None,
                        help="matrix.json listing required `runs` (config-shape subdir names)")
    parser.add_argument("--out-md", default="matrix-summary.md")
    parser.add_argument("--out-json", default="matrix-summary.json")
    args = parser.parse_args(argv)

    if not args.parent.is_dir():
        print(f"[matrix] ERROR: not a directory: {args.parent}", file=sys.stderr)
        return 2

    subdirs = sorted(
        p for p in args.parent.iterdir()
        if p.is_dir() and any(p.glob("benchmark-*.json"))
    )
    if not subdirs:
        print(f"[matrix] ERROR: no benchmark subdirectories under {args.parent}", file=sys.stderr)
        return 2

    loaded = [load_config_dir(p) for p in subdirs]

    overall_ok = all(info["ok"] for info in loaded)

    # Fail-closed completeness against an expected run list, if provided.
    expected_missing: list[str] = []
    if args.expect is not None:
        try:
            spec = json.loads(args.expect.read_text(encoding="utf-8"))
            required = spec.get("runs", [])
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[matrix] ERROR: cannot read {args.expect}: {exc}", file=sys.stderr)
            return 2
        present_names = {info["dir"] for info in loaded}
        expected_missing = [name for name in required if name not in present_names]
        if expected_missing:
            overall_ok = False

    # Group by shape.
    shapes: dict[str, dict[str, dict]] = {}
    for info in loaded:
        shape = info["shape"] or "unknown"
        config = info["config"] or info["dir"]
        shapes.setdefault(shape, {})[config] = info

    markdown = build_markdown(shapes, overall_ok)
    if expected_missing:
        markdown += (
            "\n> MISSING required runs: "
            + ", ".join(f"`{name}`" for name in expected_missing)
            + "\n"
        )

    summary = {
        "matrix_ok": overall_ok,
        "expected_missing": expected_missing,
        "configs": [
            {
                "dir": info["dir"],
                "config": info["config"],
                "shape": info["shape"],
                "engine": info["engine"],
                "precision": info["precision"],
                "ok": info["ok"],
                "reasons": info["reasons"],
                "points": info["points"],
            }
            for info in loaded
        ],
    }

    out_md = args.parent / args.out_md
    out_json = args.parent / args.out_json
    try:
        out_md.write_text(markdown, encoding="utf-8")
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[matrix] ERROR: cannot write report: {exc}", file=sys.stderr)
        return 2

    print(markdown, end="")
    print(f"\n[matrix] wrote {out_md} and {out_json}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
