# Concurrency benchmark summary

Run status: **PASS**.

vllm-dp2 / balanced / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 4082 | 68.0 | 0.073 | 0.081 | 69.1 | 3.77 | 3.78 | 0 | 23 | 42272 | 15/0 |
| 8 | PASS | 30789 | 513.2 | 0.133 | 0.161 | 66.5 | 3.97 | 4.01 | 93 | 276 | 42272 | 120/0 |
| 16 | PASS | 57527 | 958.8 | 0.187 | 0.292 | 63.3 | 4.23 | 4.35 | 93 | 272 | 42272 | 240/0 |
| 32 | PASS | 103340 | 1722.3 | 0.245 | 0.501 | 57.7 | 4.68 | 4.84 | 92 | 276 | 42276 | 480/0 |
| 64 | PASS | 166413 | 2773.6 | 0.320 | 0.728 | 48.4 | 5.59 | 6.12 | 91 | 276 | 42276 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 68.0 | 1.0x | 68.0 |
| 8 | 513.2 | 7.5x | 64.1 |
| 16 | 958.8 | 14.1x | 59.9 |
| 32 | 1722.3 | 25.3x | 53.8 |
| 64 | 2773.6 | 40.8x | 43.3 |
