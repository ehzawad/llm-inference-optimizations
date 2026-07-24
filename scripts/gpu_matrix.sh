#!/usr/bin/env bash
# Multi-GPU comparative inference benchmark driver (vLLM + llama.cpp).
#
# ONE config x ONE shape per invocation. Starts a fixed, production-style server
# (or two replicas for data-parallel), sweeps CLIENT concurrency with the
# hardened external HTTP client on the Python 3.14 harness venv, and reports
# fail-closed. Prefill (TTFT) and decode (steady-state tok/s) are captured
# separately by the client. GPU pinning is explicit and single-card configs
# leave the OTHER card completely invisible to the server process.
#
#   Usage: gpu_matrix.sh <config> <parent_run_dir> [shape]
#     config: llamacpp-a5000 | llamacpp-a6000 | vllm-a5000 | vllm-a6000
#             | vllm-tp2 | vllm-dp2
#     shape:  balanced (default) | prefill | decode
#
#   Env overrides:
#     CLIENTS   concurrency points          (default depends on shape)
#     MEASURED  measured requests / worker   (default 15)
#     WARMUP    warmup requests / worker     (default 2)
#     PORT      base server port             (default 8400; dp2 also uses PORT+1)
#     GGUF      llama.cpp gguf               (default Q6_K)
#     MERGED    vLLM merged weights dir      (default models/merged)
#     GPU_MEM_UTIL vLLM --gpu-memory-utilization (default 0.85)
#     DRY_RUN=1 print the resolved plan + exact commands, launch NOTHING
#
# The orchestrator (not this script's author) runs GPU work, strictly serialized.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# device 0 == physical A5000, device 1 == physical A6000 (matches nvidia-smi -i).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CONFIG="${1:?usage: gpu_matrix.sh <config> <parent_run_dir> [shape]}"
PARENT="${2:?usage: gpu_matrix.sh <config> <parent_run_dir> [shape]}"
SHAPE="${3:-balanced}"

HARNESS_PY="$ROOT/.venv-harness/bin/python"
VLLM="$ROOT/.venv-vllm-stable/bin/vllm"
BIN="$ROOT/vendor/llama.cpp/build/bin"
GGUF="${GGUF:-$ROOT/models/gguf/qwen3-4b-legal-ops-Q6_K.gguf}"
MERGED="${MERGED:-$ROOT/models/merged}"
ALIAS="qwen3-4b-legal-q6k"
PORT="${PORT:-8400}"
MEASURED="${MEASURED:-15}"; WARMUP="${WARMUP:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
DRY_RUN="${DRY_RUN:-0}"

# ---- placement matrix ------------------------------------------------------
# ENGINE, DEVS (per-replica CUDA_VISIBLE_DEVICES), GPUIDX (physical nvidia-smi
# indices to telemeter), TP (tensor-parallel size), REPLICAS (data-parallel).
case "$CONFIG" in
  llamacpp-a5000) ENGINE=llamacpp; DEVS=("0");     GPUIDX="0";   TP=1; REPLICAS=1; PREC="Q6_K";;
  llamacpp-a6000) ENGINE=llamacpp; DEVS=("1");     GPUIDX="1";   TP=1; REPLICAS=1; PREC="Q6_K";;
  vllm-a5000)     ENGINE=vllm;     DEVS=("0");     GPUIDX="0";   TP=1; REPLICAS=1; PREC="bf16";;
  vllm-a6000)     ENGINE=vllm;     DEVS=("1");     GPUIDX="1";   TP=1; REPLICAS=1; PREC="bf16";;
  vllm-tp2)       ENGINE=vllm;     DEVS=("0,1");   GPUIDX="0,1"; TP=2; REPLICAS=1; PREC="bf16";;
  vllm-dp2)       ENGINE=vllm;     DEVS=("0" "1"); GPUIDX="0,1"; TP=1; REPLICAS=2; PREC="bf16";;
  *) echo "unknown config: $CONFIG" >&2; exit 2;;
esac

# ---- shape presets ---------------------------------------------------------
case "$SHAPE" in
  balanced) PROMPTS="$ROOT/prompts/short-chat.jsonl";   GEN=256; SLOTCTX=1024; DEFCLIENTS="1 8 16 32 64";;
  prefill)  PROMPTS="$ROOT/prompts/long-prompt.jsonl";  GEN=32;  SLOTCTX=2048; DEFCLIENTS="1 4 8 16";;
  decode)   PROMPTS="$ROOT/prompts/short-prompt.jsonl"; GEN=512; SLOTCTX=1024; DEFCLIENTS="1 8 16 32 64";;
  *) echo "unknown shape: $SHAPE (balanced|prefill|decode)" >&2; exit 2;;
esac
CLIENTS="${CLIENTS:-$DEFCLIENTS}"

# Fixed-server capacity = the largest concurrency in the sweep.
NP=1; for c in $CLIENTS; do (( c > NP )) && NP=$c; done
LLAMA_CTX=$(( NP * SLOTCTX ))     # llama.cpp preallocates KV per slot
VLLM_MAXLEN="$SLOTCTX"            # vLLM paged KV: per-request length

RUN="$PARENT/$CONFIG-$SHAPE"
mkdir -p "$RUN"

# Client URLs (one per replica) and matching ports.
PORTS=(); URLS=()
for r in $(seq 0 $(( REPLICAS - 1 ))); do
  p=$(( PORT + r )); PORTS+=("$p"); URLS+=("http://127.0.0.1:$p")
done
URL_ARGS=(); for u in "${URLS[@]}"; do URL_ARGS+=(--url "$u"); done

expected_tags_json(){ local first=1; printf '['; for c in $CLIENTS; do
  [ $first -eq 1 ] || printf ', '; first=0; printf '"c%03d"' "$c"; done; printf ']'; }
points_json(){ local first=1; printf '['; for c in $CLIENTS; do
  [ $first -eq 1 ] || printf ', '; first=0; printf '%d' "$c"; done; printf ']'; }

# ---- server launch commands (as strings, for both DRY_RUN and real run) ----
llama_cmd(){ # $1=cuda_visible $2=port
  printf 'CUDA_VISIBLE_DEVICES=%s setsid %s -m %s --alias %s --host 127.0.0.1 --port %s -dev CUDA0 -sm none --main-gpu 0 -ngl all --fit off -fa on -np %s --ctx-size %s --no-kv-unified -cb -b 2048 -ub 512 -ctk f16 -ctv f16 --cache-ram 0 --no-cache-prompt --no-context-shift -rea off --jinja --no-webui --metrics --slots' \
    "$1" "$BIN/llama-server" "$GGUF" "$ALIAS" "$2" "$NP" "$LLAMA_CTX"
}
vllm_cmd(){ # $1=cuda_visible $2=port
  printf 'CUDA_VISIBLE_DEVICES=%s setsid %s serve %s --served-model-name %s --host 127.0.0.1 --port %s --dtype bfloat16 --tensor-parallel-size %s --max-model-len %s --max-num-seqs %s --gpu-memory-utilization %s --no-enable-prefix-caching --disable-log-requests' \
    "$1" "$VLLM" "$MERGED" "$ALIAS" "$2" "$TP" "$VLLM_MAXLEN" "$NP" "$GPU_MEM_UTIL"
}
server_cmd(){ # $1=cuda_visible $2=port
  if [ "$ENGINE" = llamacpp ]; then llama_cmd "$1" "$2"; else vllm_cmd "$1" "$2"; fi
}
client_cmd(){ # $1=concurrency $2=tag
  printf '%s scripts/bench_external.py' "$HARNESS_PY"
  for u in "${URLS[@]}"; do printf ' --url %s' "$u"; done
  printf ' --engine %s --placement %s --concurrency %s --outdir %s --tag %s --prompts %s --measured %s --warmup %s --gen-tokens %s --gpu-index %s' \
    "$ENGINE" "$CONFIG" "$1" "$RUN" "$2" "$PROMPTS" "$MEASURED" "$WARMUP" "$GEN" "$GPUIDX"
}

print_plan(){
  echo "== gpu_matrix plan =="
  echo "config=$CONFIG shape=$SHAPE engine=$ENGINE precision=$PREC"
  echo "gpu_telemetry=$GPUIDX tensor_parallel=$TP replicas=$REPLICAS"
  echo "prompts=$PROMPTS gen_tokens=$GEN per_slot_ctx=$SLOTCTX NP(np/max-seqs)=$NP"
  echo "clients=[$CLIENTS] measured=$MEASURED warmup=$WARMUP run=$RUN"
  echo "-- server(s) --"
  for r in $(seq 0 $(( REPLICAS - 1 ))); do
    echo "  [replica $r on CUDA_VISIBLE_DEVICES=${DEVS[$r]} port ${PORTS[$r]}]"
    echo "    $(server_cmd "${DEVS[$r]}" "${PORTS[$r]}") > $RUN/server-r$r.log 2>&1 &"
  done
  echo "-- client sweep --"
  for c in $CLIENTS; do echo "    $(client_cmd "$c" "$(printf 'c%03d' "$c")")"; done
  echo "-- report --"
  echo "    $HARNESS_PY scripts/report.py $RUN --label \"$CONFIG / $SHAPE / $ENGINE $PREC\""
}

if [ "$DRY_RUN" = 1 ]; then
  print_plan
  exit 0
fi

# ===========================================================================
# Live execution (orchestrator only).
# ===========================================================================
fail=0
ACTIVE_PIDS=()

if ! "$HARNESS_PY" scripts/capture_env.py "$RUN/manifest.json" >/dev/null; then
  echo "### environment capture FAILED; continuing ###" >&2
  fail=1
fi

cat > "$RUN/experiment.json" <<JSON
{
  "benchmark": {
    "concurrency_points": $(points_json),
    "expected_tags": $(expected_tags_json),
    "measured_per_worker": $MEASURED
  }
}
JSON
cat > "$RUN/config.json" <<JSON
{
  "config": "$CONFIG", "engine": "$ENGINE", "shape": "$SHAPE",
  "precision": "$PREC", "gen_tokens": $GEN, "per_slot_ctx": $SLOTCTX,
  "tensor_parallel": $TP, "replicas": $REPLICAS,
  "gpu_telemetry_indices": "$GPUIDX", "np_or_max_seqs": $NP,
  "ports": "${PORTS[*]}", "measured_per_worker": $MEASURED, "warmup_per_worker": $WARMUP
}
JSON

free_ports(){ local p; for p in "${PORTS[@]}"; do fuser -k "${p}/tcp" 2>/dev/null || true; done; sleep 3; }
wait_health(){ # $1=pid $2=port $3=timeout
  local pid="$1" port="$2" timeout="${3:-600}" t0=$SECONDS ep
  while (( SECONDS-t0 < timeout )); do
    kill -0 "$pid" 2>/dev/null || return 1
    for ep in /health /v1/models; do
      curl -sf "http://127.0.0.1:$port$ep" >/dev/null 2>&1 && return 0
    done
    sleep 2
  done
  return 1
}
stop_group(){
  local pid="${1:-}" target="" pgid=""
  [ -n "$pid" ] || return 0
  # The session leader can exit before its worker children, and under job
  # control (`set -m`) setsid's async child is already a group leader while $!
  # names a short-lived parent -- so $! is not necessarily the PGID. Prefer the
  # process's real PGID; fall back to the pid (the common non-interactive
  # setsid case, and when the leader has already exited).
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null; then
    target="-$pgid"
  elif kill -0 -- "-$pid" 2>/dev/null; then
    target="-$pid"
  elif kill -0 "$pid" 2>/dev/null; then
    target="$pid"
  fi
  if [ -n "$target" ]; then
    kill -TERM -- "$target" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 -- "$target" 2>/dev/null || break; sleep 1; done
    kill -0 -- "$target" 2>/dev/null && kill -KILL -- "$target" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
stop_all(){ local p; for p in "${ACTIVE_PIDS[@]:-}"; do [ -n "$p" ] && stop_group "$p"; done; ACTIVE_PIDS=(); }
cleanup(){ stop_all; }
on_signal(){ trap - EXIT INT TERM; cleanup; exit 130; }
trap cleanup EXIT
trap on_signal INT TERM

launch_servers(){
  local r cmd
  for r in $(seq 0 $(( REPLICAS - 1 ))); do
    cmd=$(server_cmd "${DEVS[$r]}" "${PORTS[$r]}")
    echo "### launch replica $r: $cmd ###"
    eval "$cmd > \"$RUN/server-r$r.log\" 2>&1 &"
    ACTIVE_PIDS+=("$!")
  done
  local i pid
  for i in $(seq 0 $(( REPLICAS - 1 ))); do
    pid="${ACTIVE_PIDS[$i]}"
    if ! wait_health "$pid" "${PORTS[$i]}" 900; then
      echo "### replica $i never became ready (see server-r$i.log) ###" >&2
      return 1
    fi
  done
  return 0
}

echo "### MATRIX $CONFIG/$SHAPE  clients=[$CLIENTS] $(date -u +%H:%M:%S) ###"
free_ports
if launch_servers; then
  for c in $CLIENTS; do
    tag=$(printf 'c%03d' "$c")
    if ! eval "$(client_cmd "$c" "$tag")"; then
      echo "### $CONFIG/$SHAPE c$c FAILED ###" >&2
      fail=1
    fi
  done
else
  fail=1
fi
stop_all
free_ports

if ! "$HARNESS_PY" scripts/report.py "$RUN" --label "$CONFIG / $SHAPE / $ENGINE $PREC"; then
  fail=1
fi

echo "### MATRIX DONE ($RUN) fail=$fail ###"
exit "$fail"
