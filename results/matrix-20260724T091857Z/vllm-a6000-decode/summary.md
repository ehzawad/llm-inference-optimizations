# Concurrency benchmark summary

Run status: **PASS**.

vllm-a6000 / decode / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 3648 | 60.8 | 0.053 | 0.058 | 61.1 | 8.42 | 8.66 | 98 | 267 | 42272 | 15/0 |
| 8 | PASS | 28008 | 466.8 | 0.082 | 0.166 | 58.8 | 8.77 | 8.96 | 97 | 266 | 42276 | 120/0 |
| 16 | PASS | 52280 | 871.3 | 0.120 | 0.256 | 55.2 | 9.36 | 9.54 | 96 | 271 | 42278 | 240/0 |
| 32 | PASS | 104691 | 1744.9 | 0.112 | 0.448 | 55.8 | 9.26 | 9.64 | 90 | 285 | 42276 | 480/0 |
| 64 | PASS | 171156 | 2852.6 | 0.161 | 0.750 | 45.8 | 11.31 | 11.86 | 88 | 293 | 42276 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 60.8 | 1.0x | 60.8 |
| 8 | 466.8 | 7.7x | 58.3 |
| 16 | 871.3 | 14.3x | 54.5 |
| 32 | 1744.9 | 28.7x | 54.5 |
| 64 | 2852.6 | 46.9x | 44.6 |
