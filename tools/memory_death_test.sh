#!/usr/bin/env bash
# ── Memory Death Test ─────────────────────────────────────────────────────────
# Sweeps num_envs across a geometric ladder on the current GPU.
# For each config:
#   1. Patch the config with the new num_envs / minibatch_size
#   2. Launch training in the background
#   3. Watch runs/.../metrics.csv — first data row = JIT done
#   4. Sample training/sps for SAMPLE_SECONDS (default 600 = 10 min)
#   5. Kill the process, compute stats, move to next size
#
# Usage:
#   bash tools/memory_death_test.sh [SAMPLE_SECONDS]
#
# Output:
#   results/memory_death_test_<timestamp>/summary.csv
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SAMPLE_SECONDS=${1:-600}          # 10 minutes default
POLL_INTERVAL=10                  # seconds between CSV checks
ROLLOUT_LENGTH=128
NUM_MINIBATCHES=8                 # kept constant across all configs
SEED=99
BASE_CONFIG="configs/ppo_memory_death_test.json"
ENV_COUNTS=(256 1024 2048 4096 8192)

RESULTS_DIR="results/memory_death_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
echo "num_envs,minibatch_size,mean_sps,min_sps,max_sps,n_samples,jit_seconds,status" \
    > "$SUMMARY_CSV"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# ── Patch num_envs + minibatch_size into a temp copy of the config ─────────
patch_config() {
    local num_envs=$1
    local minibatch_size=$(( num_envs * ROLLOUT_LENGTH / NUM_MINIBATCHES ))
    python3 - <<PYEOF
import json, pathlib
cfg = json.loads(pathlib.Path("$BASE_CONFIG").read_text())
cfg["ppo"]["num_envs"]       = $num_envs
cfg["ppo"]["minibatch_size"] = $minibatch_size
pathlib.Path("/tmp/mdt_config.json").write_text(json.dumps(cfg, indent=2))
print(f"  num_envs={$num_envs}  minibatch_size={minibatch_size}  num_minibatches=$NUM_MINIBATCHES")
PYEOF
}

# ── Run one benchmark point ─────────────────────────────────────────────────
run_benchmark() {
    local num_envs=$1
    local minibatch_size=$(( num_envs * ROLLOUT_LENGTH / NUM_MINIBATCHES ))
    local metrics_csv="runs/hexapod_memory_death_test/brax/flat/seed_${SEED}/metrics.csv"
    local run_log="$RESULTS_DIR/log_envs${num_envs}.txt"
    local sps_log="$RESULTS_DIR/sps_envs${num_envs}.txt"
    local status="ok"

    log "════════════════════════════════════════"
    log "  num_envs=$num_envs   minibatch=$minibatch_size"
    log "════════════════════════════════════════"

    patch_config "$num_envs"

    # Wipe stale metrics from any prior run at this seed
    rm -f "$metrics_csv"

    # Launch training — inherit GPU/GL env vars from the calling shell.
    # Set them before running this script (see README / commands below).
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    python -m rl.cli \
        --config /tmp/mdt_config.json \
        --backend brax --terrain flat \
        > "$run_log" 2>&1 &
    local pid=$!
    log "Training PID: $pid"

    # ── Wait for JIT: metrics.csv must have ≥2 lines (header + first data) ──
    local jit_start jit_seconds
    jit_start=$(date +%s)
    log "Waiting for JIT compilation..."
    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "FATAL: process died during JIT at num_envs=$num_envs"
            if grep -qiE "out.of.memory|oom|cuda error|device-side assert" \
                    "$run_log" 2>/dev/null; then
                status="OOM"
            else
                status="CRASH_JIT"
            fi
            jit_seconds=$(( $(date +%s) - jit_start ))
            echo "$num_envs,$minibatch_size,,,,0,$jit_seconds,$status" >> "$SUMMARY_CSV"
            log "  → Stopping sweep at num_envs=$num_envs (status=$status)"
            return 1   # signal caller to stop
        fi
        if [[ -f "$metrics_csv" ]] && (( $(wc -l < "$metrics_csv") >= 2 )); then
            break
        fi
        sleep "$POLL_INTERVAL"
    done
    jit_seconds=$(( $(date +%s) - jit_start ))
    log "JIT done in ${jit_seconds}s — starting ${SAMPLE_SECONDS}s sampling window."

    # ── Sample training/sps from CSV for SAMPLE_SECONDS ─────────────────────
    local sample_end prev_line curr_line remaining
    sample_end=$(( $(date +%s) + SAMPLE_SECONDS ))
    prev_line=1   # skip header
    > "$sps_log"

    while (( $(date +%s) < sample_end )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "Process died during sampling."
            status="CRASH_SAMPLE"
            break
        fi
        curr_line=$(wc -l < "$metrics_csv")
        if (( curr_line > prev_line )); then
            # Pull only NEW lines, grep for training/sps, extract value field
            sed -n "$((prev_line + 1)),${curr_line}p" "$metrics_csv" \
                | grep "training/sps" \
                | awk -F',' '{print $4}' \
                >> "$sps_log" || true
            prev_line=$curr_line
        fi
        remaining=$(( sample_end - $(date +%s) ))
        log "  Sampling… ${remaining}s left | $(wc -l < "$sps_log") SPS points so far"
        sleep "$POLL_INTERVAL"
    done

    # Kill training
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    log "Killed training for num_envs=$num_envs"

    # ── Compute + print stats ────────────────────────────────────────────────
    python3 - <<PYEOF | tee -a "$SUMMARY_CSV"
import pathlib, statistics
raw = pathlib.Path("$sps_log").read_text().split()
vals = [float(x) for x in raw if x.strip()]
if not vals:
    print("$num_envs,$minibatch_size,,,,0,$jit_seconds,$status")
else:
    mean_v = statistics.mean(vals)
    min_v  = min(vals)
    max_v  = max(vals)
    print(
        f"$num_envs,$minibatch_size,"
        f"{mean_v:.1f},{min_v:.1f},{max_v:.1f},{len(vals)},$jit_seconds,$status"
    )
PYEOF
    log "Done. Cooling down 15s..."
    sleep 15
}

# ── Main sweep ───────────────────────────────────────────────────────────────
log "Memory Death Test — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log "Results dir: $RESULTS_DIR"
log "Sample window: ${SAMPLE_SECONDS}s per config"

for num_envs in "${ENV_COUNTS[@]}"; do
    run_benchmark "$num_envs" || break   # stop sweep on OOM/crash
done

log "════════════════════════════════════════"
log "  SWEEP COMPLETE"
log "════════════════════════════════════════"
echo ""
cat "$SUMMARY_CSV"
log "Full results → $RESULTS_DIR/summary.csv"
