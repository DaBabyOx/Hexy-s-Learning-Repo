#!/usr/bin/env bash
set -euo pipefail

BASELINE_CONFIG=${BASELINE_CONFIG:-configs/ppo_baseline.json}
TUNED_CONFIG=${TUNED_CONFIG:-configs/ppo_vendor_tuned.json}
BACKENDS_BASELINE=${BACKENDS_BASELINE:-"mujoco brax"}
BACKENDS_UNEVEN=${BACKENDS_UNEVEN:-"mujoco brax"}
TUNED_BACKEND=${TUNED_BACKEND:-brax}
TERRAIN_FLAT=${TERRAIN_FLAT:-flat}
TERRAIN_UNEVEN=${TERRAIN_UNEVEN:-heightfield}

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

run_stage() {
  local stage=$1
  shift
  log "=== ${stage} ==="
  "$@"
}

run_baseline_flat() {
  for backend in ${BACKENDS_BASELINE}; do
    log "Baseline flat: backend=${backend}"
    python -m rl.cli --config "${BASELINE_CONFIG}" --backend "${backend}" --terrain "${TERRAIN_FLAT}"
  done
}

run_tuned() {
  log "Vendor-tuned: backend=${TUNED_BACKEND}"
  python -m rl.cli --config "${TUNED_CONFIG}" --backend "${TUNED_BACKEND}" --terrain "${TERRAIN_FLAT}"
}

run_baseline_uneven() {
  for backend in ${BACKENDS_UNEVEN}; do
    log "Baseline uneven: backend=${backend}"
    python -m rl.cli --config "${BASELINE_CONFIG}" --backend "${backend}" --terrain "${TERRAIN_UNEVEN}"
  done
}

run_stage "Stage 1/3: baseline flat" run_baseline_flat
run_stage "Stage 2/3: vendor-tuned" run_tuned
run_stage "Stage 3/3: baseline uneven" run_baseline_uneven

log "All stages complete."
