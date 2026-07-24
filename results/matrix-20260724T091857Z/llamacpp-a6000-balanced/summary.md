# Concurrency benchmark summary

Run status: **PASS**.

llamacpp-a6000 / balanced / llamacpp Q6_K

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 6695 | 111.6 | 0.069 | 0.075 | 114.0 | 2.31 | 2.32 | 95 | 291 | 13118 | 15/0 |
| 8 | PASS | 24385 | 406.4 | 0.410 | 0.434 | 54.9 | 5.04 | 5.12 | 78 | 281 | 13122 | 120/0 |
| 16 | PASS | 31656 | 527.6 | 0.532 | 0.744 | 41.8 | 6.62 | 11.05 | 67 | 241 | 13124 | 240/0 |
| 32 | PASS | 40547 | 675.8 | 0.732 | 0.937 | 20.0 | 13.49 | 14.59 | 56 | 232 | 13124 | 480/0 |
| 64 | PASS | 51117 | 851.9 | 0.629 | 1.199 | 13.7 | 19.23 | 19.66 | 43 | 194 | 13122 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 111.6 | 1.0x | 111.6 |
| 8 | 406.4 | 3.6x | 50.8 |
| 16 | 527.6 | 4.7x | 33.0 |
| 32 | 675.8 | 6.1x | 21.1 |
| 64 | 851.9 | 7.6x | 13.3 |
