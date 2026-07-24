# Concurrency benchmark summary

Run status: **PASS**.

vllm-tp2 / decode / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 6467 | 107.8 | 0.048 | 0.054 | 108.7 | 4.75 | 4.76 | 91 | 249 | 42422 | 15/0 |
| 8 | PASS | 45341 | 755.7 | 0.069 | 0.167 | 95.8 | 5.41 | 5.49 | 89 | 239 | 42428 | 120/0 |
| 16 | PASS | 80923 | 1348.7 | 0.164 | 0.291 | 87.1 | 6.02 | 6.23 | 88 | 240 | 42428 | 240/0 |
| 32 | PASS | 119880 | 1998.0 | 0.095 | 0.442 | 62.8 | 8.22 | 8.61 | 90 | 230 | 42428 | 480/0 |
| 64 | PASS | 166510 | 2775.2 | 0.151 | 0.746 | 44.3 | 11.67 | 12.37 | 91 | 229 | 42428 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 107.8 | 1.0x | 107.8 |
| 8 | 755.7 | 7.0x | 94.5 |
| 16 | 1348.7 | 12.5x | 84.3 |
| 32 | 1998.0 | 18.5x | 62.4 |
| 64 | 2775.2 | 25.7x | 43.4 |
