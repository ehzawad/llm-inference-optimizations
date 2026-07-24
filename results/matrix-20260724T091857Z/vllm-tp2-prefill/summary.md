# Concurrency benchmark summary

Run status: **PASS**.

vllm-tp2 / prefill / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 3656 | 60.9 | 0.207 | 0.215 | 97.4 | 0.52 | 0.53 | 89 | 197 | 42282 | 15/0 |
| 4 | PASS | 7917 | 132.0 | 0.584 | 0.629 | 85.6 | 0.98 | 1.02 | 98 | 223 | 42322 | 60/0 |
| 8 | PASS | 9861 | 164.3 | 0.827 | 1.144 | 46.0 | 1.56 | 1.58 | 98 | 236 | 42322 | 120/0 |
| 16 | PASS | 11476 | 191.3 | 1.125 | 1.911 | 20.4 | 2.66 | 3.09 | 98 | 243 | 42322 | 240/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 60.9 | 1.0x | 60.9 |
| 4 | 132.0 | 2.2x | 33.0 |
| 8 | 164.3 | 2.7x | 20.5 |
| 16 | 191.3 | 3.1x | 12.0 |
