# Concurrency benchmark summary

Run status: **PASS**.

vllm-a6000 / prefill / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 3195 | 53.2 | 0.148 | 0.151 | 68.4 | 0.60 | 0.60 | 94 | 284 | 42116 | 15/0 |
| 4 | PASS | 8285 | 138.1 | 0.397 | 0.449 | 54.8 | 0.93 | 0.96 | 94 | 279 | 42192 | 60/0 |
| 8 | PASS | 11480 | 191.3 | 0.556 | 0.820 | 40.8 | 1.33 | 1.45 | 94 | 290 | 42192 | 120/0 |
| 16 | PASS | 13960 | 232.7 | 0.678 | 1.167 | 20.7 | 2.18 | 2.51 | 97 | 294 | 42192 | 240/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 53.2 | 1.0x | 53.2 |
| 4 | 138.1 | 2.6x | 34.5 |
| 8 | 191.3 | 3.6x | 23.9 |
| 16 | 232.7 | 4.4x | 14.5 |
