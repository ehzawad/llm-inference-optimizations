# Concurrency benchmark summary

Run status: **PASS**.

vllm-a6000 / balanced / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 4108 | 68.5 | 0.066 | 0.077 | 69.6 | 3.73 | 3.78 | 94 | 287 | 42274 | 15/0 |
| 8 | PASS | 29842 | 497.4 | 0.138 | 0.276 | 64.3 | 4.11 | 4.19 | 94 | 275 | 42274 | 120/0 |
| 16 | PASS | 53520 | 892.0 | 0.234 | 0.427 | 58.5 | 4.56 | 4.76 | 92 | 279 | 42274 | 240/0 |
| 32 | PASS | 89504 | 1491.7 | 0.271 | 0.745 | 49.2 | 5.45 | 5.76 | 90 | 283 | 42278 | 480/0 |
| 64 | PASS | 134855 | 2247.6 | 0.376 | 0.956 | 37.3 | 7.29 | 7.72 | 88 | 290 | 42278 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 68.5 | 1.0x | 68.5 |
| 8 | 497.4 | 7.3x | 62.2 |
| 16 | 892.0 | 13.0x | 55.7 |
| 32 | 1491.7 | 21.8x | 46.6 |
| 64 | 2247.6 | 32.8x | 35.1 |
