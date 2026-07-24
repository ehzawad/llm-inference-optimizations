# Concurrency benchmark summary

Run status: **PASS**.

vllm-dp2 / decode / vllm bf16

Throughput = `60 * sum(completion_tokens) / makespan` (single wall clock).

| Concurrency | Status | Output tok/min | Output tok/s | TTFT p50 (s) | TTFT p95 (s) | Decode tok/s p50 | Latency p50 (s) | Latency p95 (s) | GPU util med % | Power med W | VRAM peak MiB | OK/Fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS | 4100 | 68.3 | 0.053 | 0.060 | 68.8 | 7.48 | 7.53 | 0 | 24 | 42272 | 15/0 |
| 8 | PASS | 31708 | 528.5 | 0.086 | 0.106 | 67.0 | 7.72 | 7.76 | 94 | 271 | 42272 | 120/0 |
| 16 | PASS | 60968 | 1016.1 | 0.104 | 0.205 | 64.7 | 8.00 | 8.10 | 93 | 269 | 42276 | 240/0 |
| 32 | PASS | 114027 | 1900.4 | 0.137 | 0.258 | 60.8 | 8.54 | 8.68 | 91 | 274 | 42276 | 480/0 |
| 64 | PASS | 195373 | 3256.2 | 0.121 | 0.432 | 54.4 | 9.64 | 10.20 | 90 | 278 | 42276 | 960/0 |

## Throughput scaling vs single stream

| Concurrency | Aggregate tok/s | Speedup vs C=1 | Per-stream tok/s |
|---|---|---|---|
| 1 | 68.3 | 1.0x | 68.3 |
| 8 | 528.5 | 7.7x | 66.1 |
| 16 | 1016.1 | 14.9x | 63.5 |
| 32 | 1900.4 | 27.8x | 59.4 |
| 64 | 3256.2 | 47.6x | 50.9 |
