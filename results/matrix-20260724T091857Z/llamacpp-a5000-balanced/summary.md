# Concurrency benchmark summary

Run status: **PASS**.

llamacpp-a5000 / balanced / llamacpp Q6_K

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 7258 | 121.0 | 0.070 | 0.075 | 124.6 | 2.12 | 2.14 | 94 | 229 | 12700 | 15/0 |
| 8 | PASS | 22609 | 376.8 | 0.439 | 0.464 | 50.8 | 5.43 | 5.48 | 79 | 228 | 12704 | 120/0 |
| 16 | PASS | 30782 | 513.0 | 0.587 | 0.847 | 42.8 | 6.53 | 12.06 | 65 | 217 | 12704 | 240/0 |
| 32 | PASS | 37042 | 617.4 | 0.938 | 1.009 | 18.1 | 15.08 | 15.30 | 61 | 195 | 12704 | 480/0 |
| 64 | PASS | 47523 | 792.0 | 0.710 | 1.502 | 12.8 | 20.67 | 21.24 | 45 | 166 | 12704 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 121.0 | 1.0x | 121.0 |
| 8 | 376.8 | 3.1x | 47.1 |
| 16 | 513.0 | 4.2x | 32.1 |
| 32 | 617.4 | 5.1x | 19.3 |
| 64 | 792.0 | 6.5x | 12.4 |
