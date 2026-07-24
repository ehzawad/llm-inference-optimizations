# Concurrency benchmark summary

Run status: **PASS**.

vllm-a5000 / balanced / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 4076 | 67.9 | 0.072 | 0.082 | 69.0 | 3.76 | 3.79 | 93 | 229 | 21112 | 15/0 |
| 8 | PASS | 28755 | 479.2 | 0.190 | 0.349 | 62.6 | 4.25 | 4.38 | 93 | 229 | 21112 | 120/0 |
| 16 | PASS | 51034 | 850.6 | 0.334 | 0.533 | 57.0 | 4.81 | 4.90 | 91 | 228 | 21112 | 240/0 |
| 32 | PASS | 81748 | 1362.5 | 0.422 | 0.841 | 45.6 | 5.99 | 6.27 | 90 | 232 | 21112 | 480/0 |
| 64 | PASS | 118862 | 1981.0 | 0.448 | 1.504 | 32.8 | 8.23 | 9.31 | 90 | 228 | 21116 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 67.9 | 1.0x | 67.9 |
| 8 | 479.2 | 7.1x | 59.9 |
| 16 | 850.6 | 12.5x | 53.2 |
| 32 | 1362.5 | 20.1x | 42.6 |
| 64 | 1981.0 | 29.2x | 31.0 |
