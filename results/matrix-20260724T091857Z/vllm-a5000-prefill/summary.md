# Concurrency benchmark summary

Run status: **PASS**.

vllm-a5000 / prefill / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 2983 | 49.7 | 0.179 | 0.185 | 66.7 | 0.64 | 0.65 | 93 | 208 | 20956 | 15/0 |
| 4 | PASS | 7113 | 118.5 | 0.505 | 0.598 | 50.6 | 1.08 | 1.12 | 94 | 220 | 21032 | 60/0 |
| 8 | PASS | 9406 | 156.8 | 0.744 | 1.084 | 36.0 | 1.63 | 1.75 | 98 | 226 | 21032 | 120/0 |
| 16 | PASS | 10935 | 182.3 | 0.924 | 1.380 | 16.6 | 2.79 | 3.25 | 97 | 229 | 21032 | 240/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 49.7 | 1.0x | 49.7 |
| 4 | 118.5 | 2.4x | 29.6 |
| 8 | 156.8 | 3.2x | 19.6 |
| 16 | 182.3 | 3.7x | 11.4 |
