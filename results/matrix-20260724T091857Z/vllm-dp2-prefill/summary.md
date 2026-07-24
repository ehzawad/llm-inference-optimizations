# Concurrency benchmark summary

Run status: **PASS**.

vllm-dp2 / prefill / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 3021 | 50.3 | 0.174 | 0.181 | 67.2 | 0.64 | 0.64 | 4 | 83 | 42066 | 15/0 |
| 4 | PASS | 9612 | 160.2 | 0.234 | 0.304 | 57.5 | 0.79 | 0.82 | 96 | 255 | 42114 | 60/0 |
| 8 | PASS | 14028 | 233.8 | 0.448 | 0.602 | 54.5 | 1.09 | 1.11 | 96 | 259 | 42192 | 120/0 |
| 16 | PASS | 19216 | 320.3 | 0.622 | 1.072 | 35.6 | 1.53 | 1.73 | 98 | 296 | 42192 | 240/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 50.3 | 1.0x | 50.3 |
| 4 | 160.2 | 3.2x | 40.1 |
| 8 | 233.8 | 4.6x | 29.2 |
| 16 | 320.3 | 6.4x | 20.0 |
