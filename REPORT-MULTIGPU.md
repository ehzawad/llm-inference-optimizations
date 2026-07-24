# Two GPUs, One Box: A5000 vs A6000 vs Tensor-Parallel vs Data-Parallel

A hardware-specific case study of LLM inference across **one RTX A5000 (24 GB)**, **one RTX A6000 (48 GB)**, and **both together** — as tensor-parallel (TP) and as data-parallel replicas (DP) — on a single heterogeneous, **no-NVLink**, cross-NUMA machine. Companion to [`REPORT.md`](REPORT.md) (the single-A5000 field guide); this document adds the *placement/parallelism* axis and does not restate the single-card concurrency mechanics covered there (see REPORT.md §3–§5).

Model: Qwen3-4B-Instruct-2507 + legal-ops LoRA, merged. vLLM serves **bf16** (`models/merged`); llama.cpp serves **Q6_K** GGUF. Engine servers run their own venvs; the HTTP client, reporting and analysis run on **Python 3.14** (`.venv-harness`). All numbers below come from the committed run `results/matrix-20260724T091857Z/` (14/14 config×shape runs PASS, fail-closed `--expect` satisfied, server↔client token cross-check verified on every point).

> **Read the caveats (M10) before quoting any number.** This measures *this* chassis, cooling, and heterogeneous pair — not "TP" or "DP" in general.

---

## M0. Setup & topology — the heterogeneous, no-NVLink reality

One host, two different cards on a slow interconnect:

| Field | GPU0 — RTX A5000 | GPU1 — RTX A6000 |
|---|---|---|
| VRAM | 24 564 MiB | 49 140 MiB |
| Memory bandwidth | ~768 GB/s (384-bit GDDR6) | ~768 GB/s (384-bit GDDR6) |
| CUDA cores (SMs) | 8192 (64) | 10 752 (84) |
| Board power limit | 230 W | 300 W |
| Compute capability | 8.6 | 8.6 |
| PCIe (measured) | gen3 ×16 | gen3 ×16 |

`nvidia-smi topo -m` = **`SYS`**: the two cards communicate over **PCIe + the cross-NUMA CPU interconnect (UPI)** — there is **no NVLink**. GPU0 sits on NUMA node 0, GPU1 on node 1. Every run pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` + a per-server `CUDA_VISIBLE_DEVICES`, and telemetry samples each *physical* card with `nvidia-smi -i {0,1}`.

**The two design axes.** *Tensor-parallel (TP)* shards one model across both cards, exchanging activations every layer over PCIe. *Data-parallel (DP)* runs **two independent full replicas** (one per card) behind a client-side **join-shortest-queue** router — not vLLM `--data-parallel-size 2`, which locksteps the fast card to the slow one on a no-NVLink pair (confirmed in vLLM 0.11 source: DP ranks all-reduce an "any-unfinished" flag and size KV uniformly across ranks). Two independent replicas let the A6000 profile its own (larger) KV cache and pull more load.

**Definitions (all are client-observed proxies, not kernel counters).**
- *Aggregate throughput* = `Σ completion_tokens / makespan` over one global wall clock.
- *TTFT* (prefill proxy) = time to first streamed token — includes HTTP, queueing, scheduling, and prefill, not isolated prefill-kernel time.
- *Decode tok/s* (decode proxy) = median over requests of `(gen_tokens − 1)/(latency − TTFT)` — a per-request user-visible rate, **not** an aggregate GPU decode figure, so it must never be summed across cards.

Three fixed workload shapes (server booted at `NP = max(C)`, client sweeps concurrency C):

| shape | prompt | gen | per-slot ctx | C points | engines |
|---|---|---|---|---|---|
| balanced | ~256 tok | 256 | 1024 | 1, 8, 16, 32, 64 | vLLM + llama.cpp |
| prefill-heavy | ~1240 tok | 32 | 2048 | 1, 4, 8, 16 | vLLM only |
| decode-heavy | ~13 tok | 512 | 1024 | 1, 8, 16, 32, 64 | vLLM only |

---

## M1. The one-paragraph answer

On this box, **for a 4B model that already fits on either card**:

- **Data-parallel (two replicas) is the reliable way to turn two GPUs into more throughput.** DP beats the best single card in every shape and the margin **grows with concurrency** (balanced 0.99×→**1.23×** at C=64; decode 1.00×→**1.14×**; prefill 0.95×→**1.38×** at C=16), landing at **97–99 % of the load-matched two-replica ceiling**. It costs latency almost nothing and roughly **doubles power**.
- **Tensor-parallel is a low-concurrency / single-stream tool, not a throughput-under-load tool.** TP wins big at C=1 (~**1.5×** best-single, every shape) because splitting weights ~doubles weight-read bandwidth, but the advantage **erodes as concurrency rises and flips to a loss** at high C in prefill (C≥4: 0.82–0.96×) and decode (C=64: 0.97×); on balanced it fades to near-parity (1.04× at C=64). **TP always has worse TTFT** than the best single card. The popular "TP is useless without NVLink" claim is **too coarse here** — it is false at low concurrency and true-ish only under load.
- **The A6000 is not simply "the faster card."** It wins compute-bound prefill (1.07×→1.28×) and high-concurrency batched decode, but at low-to-mid concurrency in the decode-heavy shape it is **slower than the A5000** because it **thermally throttles** in this chassis.
- **For efficiency (tokens/sec per watt), a single A5000 wins.** Two-GPU configurations trade roughly half the per-watt efficiency for absolute throughput.

Everything below is the evidence, with the mechanism and the caveats.

---

## M2. Single-card baselines — A5000 vs A6000

These are the ceilings every 2-GPU number is judged against. Same ~768 GB/s bandwidth, so **decode is expected to be close**; the A6000's extra cores should help **prefill** and **batched high-concurrency** work.

**vLLM bf16, aggregate tok/s (A6000 / A5000 ratio):**

| C | balanced A5000 | A6000 | ratio | decode A5000 | A6000 | ratio | prefill A5000 | A6000 | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 67.9 | 68.5 | 1.01× | 68.6 | 60.8 | **0.89×** | 49.7 | 53.2 | 1.07× |
| 8 | 479 | 497 | 1.04× | 505 | 467 | **0.92×** | 157 | 191 | 1.22× |
| 16 | 851 | 892 | 1.05× | 958 | 871 | **0.91×** | 182 | 233 | 1.28× |
| 32 | 1363 | 1492 | 1.09× | 1630 | 1745 | 1.07× | — | — | — |
| 64 | 1981 | 2248 | 1.13× | 2503 | 2853 | 1.14× | — | — | — |

Two honest reads:
- **Prefill (compute-bound): the A6000 wins, and the win grows with load** (1.07×→1.28×) — its extra cores matter exactly where compute is the bottleneck.
- **Decode (bandwidth-bound): the A6000 is *slower* at C≤16** (0.89–0.92×) and only pulls ahead at C≥32. This is **not** a bandwidth story (the cards are equal) — it is **thermal throttling** (M6): the A6000 runs at 85–88 °C and clocks down. Same-bandwidth cards would otherwise track each other closely.

llama.cpp Q6_K single-stream (C=1) decode is **~124 tok/s (A5000) / 114 (A6000)** — nearly 2× vLLM bf16's ~68, because Q6_K moves far fewer weight bytes per token; the ordering reverses under concurrency (M8).

---

## M3. Balanced shape — DP vs TP vs single-card (the headline)

**Aggregate throughput (tok/s), placement × concurrency:**

| placement | C=1 | C=8 | C=16 | C=32 | C=64 |
|---|---|---|---|---|---|
| vllm-a5000 | 67.9 | 479 | 851 | 1363 | 1981 |
| vllm-a6000 | 68.5 | 497 | 892 | 1492 | 2248 |
| **vllm-tp2** | **106.0** | **692** | **1153** | 1710 | 2339 |
| **vllm-dp2** | 68.0¹ | 513 | 959 | 1722 | **2774** |
| *sum of singles @C* | 136 | 977 | 1743 | 2854 | 4229 |

¹ **DP C=1 is an A5000-only single stream**, not a 2-GPU result: with one request in flight the join-shortest-queue tie-break always picks replica 0 (the A5000), and live telemetry confirms GPU1 idle. Exclude it from DP trends.

**The two 2-GPU ratios:**

| C | dp2 / best-single | tp2 / best-single | dp2 / sum@(C/2) |
|---|---|---|---|
| 1 | (A5000-only) | 1.55× | — |
| 8 | 1.03× | 1.39× | — |
| 16 | 1.07× | 1.29× | 0.98× |
| 32 | 1.15× | 1.15× | 0.99× |
| 64 | **1.23×** | 1.04× | 0.97× |

Read this as a **crossover**: TP dominates at low concurrency (1.55× at C=1) but its edge decays monotonically toward parity by C=64, while DP starts at parity and climbs to 1.23×. They cross around C=32. DP delivers **97–99 % of the load-matched two-replica ceiling** (`dp2 / sum@(C/2)`), i.e. the router and co-located servers add only 1–3 % overhead — a clean two-GPU aggregate.

**Prefill vs decode within balanced** (TTFT p50 / decode-proxy p50):

| C | a5000 TTFT | a6000 TTFT | tp2 TTFT | dp2 TTFT | a6000 decode | tp2 decode | dp2 decode |
|---|---|---|---|---|---|---|---|
| 1 | 0.072 | 0.066 | 0.072 | 0.073 | 69.6 | **108.8** | 69.1 |
| 64 | 0.448 | **0.376** | 0.648 | 0.728 | 37.3 | 40.1 | 48.4 |

TP's single-stream **decode** proxy (108.8) is ~1.56× best-single — two memory buses reading half the weights each. But TP's **TTFT is the worst of any placement at high C** (0.648 s vs the A6000's 0.376 s): the per-layer all-reduce and heterogeneous sync tax time-to-first-token even when it helps steady-state decode. **"TP wins" and "TP loses" are both wrong** — TP trades TTFT for decode, and the trade only pays at low concurrency.

---

## M4. Prefill-heavy shape — where TP should have helped most (it doesn't, past C=1)

Long prompt, gen=32, C ∈ {1,4,8,16}. Prefill is compute-bound and TP adds cores, so this is TP's best theoretical case.

| C | a5000 | a6000 | tp2 | dp2 | tp2/best-single | dp2/best-single |
|---|---|---|---|---|---|---|
| 1 | 49.7 | 53.2 | 60.9 | 50.3 | **1.14×** | 0.95× |
| 4 | 118 | 138 | 132 | 160 | 0.96× | 1.16× |
| 8 | 157 | 191 | 164 | 234 | **0.86×** | 1.22× |
| 16 | 182 | 233 | 191 | 320 | **0.82×** | **1.38×** |

TP helps only at C=1 (1.14×), then **loses to a single A6000 for all C≥4** — and its TTFT is the worst on the board (1.125 s at C=16 vs the A6000's 0.678 s). Once several prefills batch on one card, the extra cores are already saturated and the cross-PCIe all-reduce is pure overhead. **DP is the clear prefill winner**, scaling to 1.38× best-single at C=16.

---

## M5. Decode-heavy shape — DP's regime, and TP's crossover to a loss

Short prompt, gen=512, C ∈ {1,8,16,32,64}.

| C | a5000 | a6000 | tp2 | dp2 | tp2/best-single | dp2/best-single |
|---|---|---|---|---|---|---|
| 1 | 68.6 | 60.8 | 107.8 | 68.3 | 1.57× | 1.00× |
| 8 | 505 | 467 | 756 | 528 | 1.50× | 1.05× |
| 16 | 958 | 871 | 1349 | 1016 | 1.41× | 1.06× |
| 32 | 1630 | 1745 | 1998 | 1900 | 1.15× | 1.09× |
| 64 | 2503 | 2853 | 2775 | **3256** | **0.97×** | **1.14×** |

This is the cleanest picture: **TP leads up to C=32 then crosses below the best single card at C=64 (0.97×)**, exactly where batched decode saturates compute and the per-token all-reduce over PCIe becomes the bottleneck. **DP overtakes it and keeps scaling (1.14×).** Use `output_tokens_per_s` for cluster throughput here — the per-request `decode_tokens_per_s` proxy is not additive across replicas and must not be summed.

---

## M6. The topology tax — per-card telemetry & the throttle audit

Using `telemetry_by_gpu` (per physical card), at C=64:

| config | card | util med | power med | temp med | sw_thermal_slowdown | sw_power_cap |
|---|---|---|---|---|---|---|
| tp2-balanced | A5000 | 87 % | 224 W | 79 °C | 0 % | **98 %** |
| tp2-balanced | A6000 | 87 % | 229 W | 84 °C | **77 %** | 0 % |
| dp2-balanced | A5000 | 91 % | 231 W | 79 °C | 0 % | **98 %** |
| dp2-balanced | A6000 | 91 % | 276 W | 87 °C | **73 %** | 24 % |
| a6000-decode | A6000 | 88 % | 293 W | 86 °C | **81 %** | 30 % |

Three findings, each corrects a tempting assumption:

1. **TP is not "one card idling in lockstep."** Both cards run ~87 % utilization in TP — the two GPUs *are* both working; TP's loss under load is a communication/synchronization tax, not idle silicon. (An earlier mechanism prediction of "A6000 idles waiting" is contradicted by this telemetry.)
2. **The two cards hit *different* limits.** The A5000 is **power-capped** (at its 230 W board limit in ~98 % of samples); the A6000 is **thermally throttled** (sw_thermal_slowdown active in 63–93 % of high-C samples, 85–88 °C). This asymmetry is *why* the A6000 underperforms in low-mid-concurrency decode (M2).
3. **DP is well balanced.** Both replicas sit at ~91 % util — JSQ genuinely spreads load; the A6000 replica draws more power (276 W vs 231 W) doing its larger share.

Mechanisms here are **consistent with** PCIe/cross-NUMA cost and this chassis's cooling; they are **not isolated** — no NCCL/PCIe bandwidth trace or NUMA-pinning ablation was run (M10).

---

## M7. Efficiency — tokens/sec per watt (board power)

Per-watt = `aggregate tok/s ÷ Σ per-card median board power`, at each config's peak-throughput C:

| shape | config | peak tok/s | Σ power | **tok/s per W** | J / 1k tok |
|---|---|---|---|---|---|
| balanced | vllm-a5000 | 1981 | 228 W | **8.68** | 115 |
| balanced | vllm-a6000 | 2248 | 290 W | 7.74 | 129 |
| balanced | vllm-tp2 | 2339 | 453 W | 5.16 | 194 |
| balanced | vllm-dp2 | 2774 | 507 W | 5.47 | 183 |
| decode | vllm-a5000 | 2503 | 228 W | **10.97** | 91 |
| decode | vllm-dp2 | 3256 | 508 W | 6.41 | 156 |

The single **A5000 is the efficiency winner** — lowest power, competitive throughput. Two-GPU configs deliver more absolute tokens/sec but at **~40 % lower efficiency** (you pay ~2× power for 1.05–1.4× throughput). DP is slightly more efficient than TP. This is **GPU-board power only** (not host/PSU/system), summed from 1 Hz `nvidia-smi` medians — an approximation, not integrated joules. Per-*dollar* is omitted: this box has no defensible price basis.

---

## M8. Cross-engine at balanced (secondary — precision-mismatched)

**Iso-*placement*, not iso-*precision*** (llama.cpp Q6_K vs vLLM bf16). This is a sanity cross-read, not a fair engine benchmark — see REPORT.md §5 for the matched-bf16 shootout.

| C | llama.cpp a5000 | vLLM a5000 | llama.cpp a6000 | vLLM a6000 |
|---|---|---|---|---|
| 1 | **121** | 68 | **112** | 68 |
| 64 | 792 | **1981** | 852 | **2248** |

llama.cpp wins single-stream (Q6_K moves fewer bytes); vLLM wins ~2.5× under concurrency. **Both engines run with continuous batching** (`llama-server -cb`), so the gap is **not** "batching vs none" — it is that vLLM's batching/kernel/attention stack scales better than this llama.cpp Q6_K configuration at high concurrency.

---

## M9. Decision guide — "I have these two cards, what do I run?"

| Goal | Best choice | Evidence |
|---|---|---|
| Max aggregate throughput under load | **DP (two replicas)** | M3/M5: 1.14–1.23× best-single at C=64, scales with C |
| Lowest single-stream / low-concurrency latency | **TP** | M3–M5: ~1.5× at C=1, all shapes |
| Best time-to-first-token under load | **single A6000** | M3: TP/DP both worsen TTFT at high C |
| Best tokens/sec per watt | **single A5000** | M7: 8.7–11.0 tok/s/W |
| Simplicity / one model, one card | **single A6000** (prefill-heavy) or **A5000** (decode/efficiency) | M2 |

**Is TP ever the right call on this box?** Yes — but narrowly: a **latency-sensitive, low-concurrency** service for a model that fits on one card, where its ~1.5× single-stream decode outweighs its worse TTFT. For anything throughput-oriented or beyond ~C=32, **DP wins**. A model that *doesn't* fit on one card would *force* TP (or a bigger card) — a regime this study did not test.

---

## M10. Limitations & caveats (read as carefully as the results)

1. **Heterogeneous pair.** A5000 (24 GB, 230 W) + A6000 (48 GB, 300 W). TP is throttled toward the more-limited card and DP is load-*balanced* by JSQ but the cards have different ceilings. A matched 2×A5000 or 2×A6000 box could move any of these numbers in *either* direction — do not extrapolate.
2. **No NVLink — `SYS` interconnect.** TP all-reduces cross PCIe gen3 + cross-NUMA UPI. Every TP number is *TP-on-SYS*; the same model over NVLink would look different. "TP loses under load" is a property of *this interconnect*, not of TP.
3. **Cross-NUMA / host contention is inferred, not isolated.** No NCCL, PCIe-bandwidth, or NUMA-pinning ablation was run. DP's 1–3 % shortfall vs the ideal ceiling is *consistent with* host/NUMA cost — not proven to be it.
4. **Router = join-shortest-queue, one policy.** DP results reflect JSQ least-connections routing. C=1 DP is A5000-only (deterministic tie-break). Per-replica request shares are corroborated by server logs but were not persisted into the benchmark JSON, so the report describes the algorithm, not a measured per-card share.
5. **Precision is not matched across engines.** Q6_K vs bf16; cross-engine cells (M8) are directional only.
6. **Closed-loop, barrier-synchronous, fixed-server/vary-client, ascending C, no reset between points.** p50/p95 reflect a saturated server, not Poisson arrivals — never compare to open-loop/Locust numbers.
7. **One 4B model, three shapes, these C points.** The model fits on one card; **capacity-forced TP was not studied**. Bigger models/longer sequences grow the TP all-reduce and would change the verdict.
8. **A6000 thermal throttling is a first-class result.** 85–88 °C with sw_thermal_slowdown active in most high-C samples. This measures *this chassis, cooling, run order, and thermal equilibrium* — not datasheet-ideal A6000 performance.
9. **Metrics are client-observed proxies.** TTFT includes HTTP/queue/schedule; decode-rate includes scheduler gaps; both are not isolated kernel times. `memory.used` reflects vLLM's reserved fraction (~0.85), not live KV. Telemetry is 1 Hz; per-watt is GPU-board power, not system energy.
10. **No independent repetitions.** C=1 has 15 measured requests (p95 ≈ near-max). Treat differences of a few percent (e.g. TP/best-single 1.04× at balanced C=64) as **near parity** unless repeated.

---

## M11. Reproduction & provenance

- Driver: `scripts/gpu_matrix.sh <config> <run_dir> <shape>` (pins GPUs, boots server, sweeps C, fail-closed report); aggregate with `scripts/matrix_report.py <run_dir> --expect config/matrix.json`. Preview any cell's exact server+client command with `DRY_RUN=1`.
- Run: `results/matrix-20260724T091857Z/` — 14/14 PASS, `matrix_ok`, `--expect` satisfied, server↔client token cross-check non-null on every point.
- Provenance: each run's `manifest.json` records commit + `source_dirty` (false) and now enumerates **both** GPUs. Raw server logs stay local; the derived `benchmark-*.json`, `summary.*`, `matrix-summary.*`, per-GPU telemetry summaries and manifests are committed as the evidence bundle.
- Committed on branch `pr2-fixes-and-multigpu`; base = the PR-#2 fail-closed hardening.
