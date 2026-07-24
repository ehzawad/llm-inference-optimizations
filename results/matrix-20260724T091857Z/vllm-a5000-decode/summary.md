# Concurrency benchmark summary

Run status: **PASS**.

vllm-a5000 / decode / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 4115 | 68.6 | 0.052 | 0.060 | 68.9 | 7.47 | 7.52 | 94 | 230 | 21112 | 15/0 |
| 8 | PASS | 30297 | 504.9 | 0.114 | 0.164 | 63.9 | 8.11 | 8.17 | 92 | 230 | 21116 | 120/0 |
| 16 | PASS | 57502 | 958.4 | 0.168 | 0.280 | 61.1 | 8.52 | 8.70 | 93 | 230 | 21116 | 240/0 |
| 32 | PASS | 97825 | 1630.4 | 0.149 | 0.523 | 52.2 | 9.92 | 10.24 | 92 | 230 | 21116 | 480/0 |
| 64 | PASS | 150168 | 2502.8 | 0.153 | 0.795 | 39.9 | 13.02 | 13.35 | 88 | 228 | 21116 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 68.6 | 1.0x | 68.6 |
| 8 | 504.9 | 7.4x | 63.1 |
| 16 | 958.4 | 14.0x | 59.9 |
| 32 | 1630.4 | 23.8x | 51.0 |
| 64 | 2502.8 | 36.5x | 39.1 |
