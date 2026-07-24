#!/usr/bin/env python3
"""Closed-loop benchmark against an ALREADY-RUNNING OpenAI-compatible server.

This harness is used for engine comparisons. It reuses benchmark.py's telemetry
and metric parsing, but sends only fields accepted by both llama.cpp and vLLM.
A run is successful only when every planned request produced a valid record and,
when available, the server's generation counter agrees with the client count.

Usage: bench_external.py --url http://127.0.0.1:8199 --engine vllm \
         --concurrency 30 --outdir results/engines/<run> --tag vllm-c030 \
         [--measured 20 --warmup 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import Telemetry, get_metrics, gpu_mem_used  # noqa: E402


BENCH_VERSION = "1.2"
GENERATION_COUNTERS = (
    "vllm:generation_tokens_total",
    "sglang:generation_tokens_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:n_tokens_predicted_total",
)


def do_request(base: str, prompt: str, max_tokens: int = 256) -> dict[str, Any]:
    """Send one streaming request and return timings plus usage."""
    payload = {
        "model": "qwen3-4b-legal-q6k",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 12345,
    }
    url = urljoin(base, "/v1/chat/completions")
    t_start = time.perf_counter()
    ttft = None
    completion_tokens = prompt_tokens = finish_reason = status = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as response:
            status = response.status_code
            if response.status_code != 200:
                return {
                    "ok": False,
                    "status": status,
                    "error": response.text[:200],
                    "t_start": t_start,
                    "t_end": time.perf_counter(),
                }
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    if (choices[0].get("delta") or {}).get("content") and ttft is None:
                        ttft = time.perf_counter() - t_start
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                if obj.get("usage"):
                    completion_tokens = obj["usage"].get("completion_tokens")
                    prompt_tokens = obj["usage"].get("prompt_tokens")
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status": status,
            "error": str(exc)[:200],
            "t_start": t_start,
            "t_end": time.perf_counter(),
        }

    t_end = time.perf_counter()
    ok = status == 200 and completion_tokens is not None and finish_reason == "length"
    return {
        "ok": ok,
        "status": status,
        "t_start": t_start,
        "t_end": t_end,
        "ttft": ttft,
        "latency": t_end - t_start,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
    }


def failed_request(exc: BaseException) -> dict[str, Any]:
    now = time.perf_counter()
    return {
        "ok": False,
        "status": None,
        "error": f"{type(exc).__name__}: {exc}"[:200],
        "t_start": now,
        "t_end": now,
        "ttft": None,
        "latency": 0.0,
        "completion_tokens": None,
        "prompt_tokens": None,
        "finish_reason": None,
    }


def run_phase(
    base: str | list[str],
    prompts: list[str],
    concurrency: int,
    per_worker: int,
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Run a barrier-synchronized closed-loop phase.

    ``base`` may be a single URL or a list of replica URLs; requests are
    round-robined across replicas so a data-parallel deployment is offered a
    balanced share of load under one wall clock. Worker exceptions are converted
    into failed request records instead of being lost on background threads, so
    the returned list contains exactly ``concurrency * per_worker`` records
    unless thread creation itself fails.
    """
    bases = [base] if isinstance(base, str) else list(base)
    counter = {"i": 0}
    counter_lock = threading.Lock()
    sink: list[dict[str, Any]] = []
    barrier = threading.Barrier(concurrency)

    def next_index() -> int:
        with counter_lock:
            i = counter["i"]
            counter["i"] += 1
        return i

    def worker() -> None:
        try:
            barrier.wait()
        except threading.BrokenBarrierError as exc:
            sink.extend(failed_request(exc) for _ in range(per_worker))
            return
        for _ in range(per_worker):
            i = next_index()
            target = bases[i % len(bases)]
            prompt = prompts[i % len(prompts)]
            try:
                result = do_request(target, prompt, max_tokens)
            except Exception as exc:  # keep thread failures visible in the artifact
                result = failed_request(exc)
            sink.append(result)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sink


def wait_ready(base: str, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for endpoint in ("/health", "/v1/models"):
            try:
                if requests.get(base.rstrip("/") + endpoint, timeout=3).status_code < 400:
                    return
            except requests.RequestException:
                pass
        time.sleep(1.0)
    raise RuntimeError("server never became ready")


def percentile(values: list[float], p: float) -> float | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else None


def prometheus_metric_total(metrics: dict[str, float], name: str) -> float | None:
    """Sum one Prometheus metric family across exact and labelled samples."""
    values = [
        value
        for key, value in metrics.items()
        if key == name or key.startswith(f"{name}{{")
    ]
    return sum(values) if values else None


def generation_counter_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> tuple[float | None, str | None]:
    """Return the first available engine generation-counter delta and its name."""
    for name in GENERATION_COUNTERS:
        before_total = prometheus_metric_total(before, name)
        after_total = prometheus_metric_total(after, name)
        if before_total is not None and after_total is not None:
            return after_total - before_total, name
    return None, None


def aggregate_generation_delta(
    befores: list[dict[str, float]],
    afters: list[dict[str, float]],
) -> tuple[float | None, str | None]:
    """Sum the generation-counter delta across one or more replica scrapes.

    For a single server this is just ``generation_counter_delta``. For a
    data-parallel deployment (one scrape per replica) the per-replica deltas are
    summed so the server-side token count still cross-checks the client total.
    Returns ``(None, None)`` only when NO replica exposed a usable counter.
    """
    total = 0.0
    names: list[str] = []
    found = False
    for before, after in zip(befores, afters):
        delta, name = generation_counter_delta(before, after)
        if delta is not None:
            total += delta
            found = True
            if name and name not in names:
                names.append(name)
    if not found:
        return None, None
    return total, "+".join(names)


def decode_rate(record: dict[str, Any]) -> float | None:
    """Per-request steady-state decode rate: (gen_tokens-1)/(latency-ttft).

    Isolates the decode phase from prefill by subtracting TTFT and the first
    token, so it is a clean decode-throughput proxy distinct from the aggregate
    system throughput (tokens/makespan). Returns None when it cannot be formed.
    """
    tokens = record.get("completion_tokens")
    ttft = record.get("ttft")
    latency = record.get("latency")
    if not isinstance(tokens, (int, float)) or tokens is None or tokens <= 1:
        return None
    if ttft is None or latency is None:
        return None
    decode_window = latency - ttft
    if decode_window <= 0:
        return None
    return (tokens - 1) / decode_window


def counter_matches(client_tokens: int, server_tokens: float | None) -> bool | None:
    if server_tokens is None:
        return None
    return abs(float(client_tokens) - float(server_tokens)) <= 0.5


def successful_run(
    requests_ok: int,
    requests_failed: int,
    expected_requests: int,
    server_counter_matches: bool | None,
) -> bool:
    return (
        requests_ok == expected_requests
        and requests_failed == 0
        and server_counter_matches is not False
    )


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    prompt = item["prompt"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid prompt record: {exc}") from exc
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
                prompts.append(prompt)
    except OSError as exc:
        raise ValueError(f"cannot read prompts from {path}: {exc}") from exc
    if not prompts:
        raise ValueError(f"prompt corpus is empty: {path}")
    return prompts


def _telemetry_peak_vram(summary: dict[str, Any] | None) -> float | None:
    mem = (summary or {}).get("mem_used_mib")
    return mem.get("peak") if isinstance(mem, dict) else None


def percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, action="append",
        help="server base URL; repeat for data-parallel replicas (round-robined)",
    )
    parser.add_argument("--engine", default="unknown")
    parser.add_argument(
        "--placement", default=None,
        help="matrix placement label recorded in the artifact (e.g. vllm-dp2)",
    )
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prompts", default="prompts/short-chat.jsonl")
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gen-tokens", type=int, default=256,
                        help="generated tokens per request (ignore_eos), for shape control")
    parser.add_argument(
        "--gpu-index", default="0",
        help="comma-separated PHYSICAL nvidia-smi GPU indices to telemeter "
             "(e.g. '0', '1', or '0,1' for two-GPU placements)",
    )
    args = parser.parse_args(argv)

    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.measured <= 0:
        parser.error("--measured must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.gen_tokens <= 0:
        parser.error("--gen-tokens must be positive")

    urls = [u.rstrip("/") + "/" for u in args.url]
    gpu_indices = [tok.strip() for tok in args.gpu_index.split(",") if tok.strip()]
    if not gpu_indices:
        gpu_indices = ["0"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / f"benchmark-{args.tag}.json"
    expected_measured = args.concurrency * args.measured
    result: dict[str, Any] = {
        "bench_version": BENCH_VERSION,
        "engine": args.engine,
        "placement": args.placement,
        "tag": args.tag,
        "urls": urls,
        "gpu_indices": gpu_indices,
        "concurrency": args.concurrency,
        "warmup_per_worker": args.warmup,
        "measured_per_worker": args.measured,
        "gen_tokens": args.gen_tokens,
        "requests_expected": expected_measured,
        "ok": False,
    }

    # Corpus/setup errors happen before any request; still emit a structured
    # failure artifact so an aborted point is never silently absent from the run.
    try:
        prompts = load_prompts(Path(args.prompts))
    except ValueError as exc:
        result.update({"ok": False, "error_type": "SetupError", "error": str(exc)})
        write_result(result_path, result)
        print(f"[ext] ERROR: {exc}", file=sys.stderr)
        print(f"[ext] wrote failure artifact {result_path}", file=sys.stderr)
        return 2

    # One telemetry sampler per involved physical GPU (both cards for tp2/dp2).
    telemetries = [
        Telemetry(str(outdir / f"telemetry-{args.tag}-gpu{idx}.csv"), device=idx)
        for idx in gpu_indices
    ]
    telemetry_started = False
    try:
        for base in urls:
            wait_ready(base)
        vram_ready = {idx: gpu_mem_used(idx) for idx in gpu_indices}

        warmup = run_phase(urls, prompts, args.concurrency, args.warmup, args.gen_tokens)
        expected_warmup = args.concurrency * args.warmup
        warmup_bad = [record for record in warmup if not record.get("ok")]
        if len(warmup) != expected_warmup or warmup_bad:
            raise RuntimeError(
                "warmup failed: "
                f"records={len(warmup)}/{expected_warmup}, failures={len(warmup_bad)}"
            )

        time.sleep(2)
        vram_idle = {idx: gpu_mem_used(idx) for idx in gpu_indices}
        metrics_before = [get_metrics(base) for base in urls]
        try:
            for telemetry in telemetries:
                telemetry.start()
            telemetry_started = True
        except Exception:
            for telemetry in telemetries:
                telemetry.stop()
            raise
        try:
            time.sleep(1)
            sink = run_phase(urls, prompts, args.concurrency, args.measured, args.gen_tokens)
            time.sleep(1)
        finally:
            if telemetry_started:
                for telemetry in telemetries:
                    telemetry.stop()
                telemetry_started = False
        metrics_after = [get_metrics(base) for base in urls]

        ok_records = [record for record in sink if record.get("ok")]
        bad_records = [record for record in sink if not record.get("ok")]
        completion_tokens = sum(record["completion_tokens"] for record in ok_records)
        makespan = (
            max(record["t_end"] for record in ok_records)
            - min(record["t_start"] for record in ok_records)
            if ok_records
            else 0.0
        )
        output_tps = completion_tokens / makespan if makespan else 0.0
        ttfts = [record["ttft"] for record in ok_records if record["ttft"] is not None]
        latencies = [record["latency"] for record in ok_records]
        decode_rates = [
            rate for rate in (decode_rate(record) for record in ok_records) if rate is not None
        ]

        server_generated, server_metric = aggregate_generation_delta(
            metrics_before, metrics_after
        )
        counters_match = counter_matches(completion_tokens, server_generated)
        run_ok = successful_run(
            len(ok_records), len(bad_records), expected_measured, counters_match
        )

        telemetry_by_gpu = {
            f"gpu{idx}": telemetry.summarize()
            for idx, telemetry in zip(gpu_indices, telemetries)
        }
        # Representative telemetry for the single-column report: the busiest card
        # by peak VRAM (falls back to the first GPU when peaks are unavailable).
        primary = max(
            telemetry_by_gpu.values(),
            key=lambda summary: _telemetry_peak_vram(summary) or -1.0,
            default={},
        ) if telemetry_by_gpu else {}

        result.update({
            "ok": run_ok,
            "requests_ok": len(ok_records),
            "requests_failed": len(bad_records),
            "completion_tokens_total": completion_tokens,
            "makespan_s": makespan,
            "output_tokens_per_s": output_tps,
            "output_tokens_per_min": output_tps * 60,
            "server_generated_tokens_delta": server_generated,
            "server_generated_tokens_metric": server_metric,
            "server_counter_matches_client": counters_match,
            "prompt_tokens_example": ok_records[0]["prompt_tokens"] if ok_records else None,
            "ttft_s": percentiles(ttfts),
            "latency_s": percentiles(latencies),
            "decode_tokens_per_s": percentiles(decode_rates),
            "vram_ready_mib": vram_ready,
            "vram_idle_mib": vram_idle,
            "telemetry": primary,
            "telemetry_by_gpu": telemetry_by_gpu,
            "failures_sample": bad_records[:5],
        })
    except Exception as exc:
        if telemetry_started:
            for telemetry in telemetries:
                telemetry.stop()
        result.update({
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        write_result(result_path, result)
        print(f"[ext] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[ext] wrote failure artifact {result_path}", file=sys.stderr)
        return 2

    write_result(result_path, result)
    print(
        f"[ext] {args.engine} [{args.placement or 'single'}] C={args.concurrency}: "
        f"{result['output_tokens_per_min']:.0f} tok/min "
        f"({result['output_tokens_per_s']:.1f} tok/s) "
        f"decode_p50={result['decode_tokens_per_s']['p50']} "
        f"ok={result['requests_ok']} fail={result['requests_failed']} "
        f"ttft_p50={result['ttft_s']['p50']}"
    )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
