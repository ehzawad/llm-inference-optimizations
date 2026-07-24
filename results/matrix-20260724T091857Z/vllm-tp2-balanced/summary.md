# Concurrency benchmark summary

Run status: **PASS**.

vllm-tp2 / balanced / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 6359 | 106.0 | 0.072 | 0.074 | 108.8 | 2.42 | 2.42 | 91 | 245 | 42424 | 15/0 |
| 8 | PASS | 41494 | 691.6 | 0.222 | 0.336 | 93.3 | 2.96 | 3.03 | 89 | 246 | 42424 | 120/0 |
| 16 | PASS | 69195 | 1153.2 | 0.265 | 0.589 | 78.5 | 3.52 | 3.65 | 88 | 242 | 42428 | 240/0 |
| 32 | PASS | 102607 | 1710.1 | 0.461 | 0.943 | 59.4 | 4.78 | 5.07 | 88 | 234 | 42428 | 480/0 |
| 64 | PASS | 140331 | 2338.9 | 0.648 | 1.565 | 40.1 | 6.99 | 7.74 | 87 | 229 | 42428 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 106.0 | 1.0x | 106.0 |
| 8 | 691.6 | 6.5x | 86.4 |
| 16 | 1153.2 | 10.9x | 72.1 |
| 32 | 1710.1 | 16.1x | 53.4 |
| 64 | 2338.9 | 22.1x | 36.5 |
