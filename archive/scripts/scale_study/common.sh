#!/usr/bin/env bash
# Shared helpers for the scale study. Source from other scripts:
#   source "$(dirname "$0")/common.sh"
set -euo pipefail

_SCALE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT_DEFAULT="$(cd "${_SCALE_DIR}/../.." && pwd)"

# Load PROTOCOL.env if present; otherwise fall back to the example defaults.
_PROTOCOL="${SCALE_PROTOCOL:-${_SCALE_DIR}/PROTOCOL.env}"
if [[ -f "${_PROTOCOL}" ]]; then
  # shellcheck disable=SC1090
  set -a; source "${_PROTOCOL}"; set +a
elif [[ -f "${_SCALE_DIR}/PROTOCOL.env.example" ]]; then
  # shellcheck disable=SC1091
  set -a; source "${_SCALE_DIR}/PROTOCOL.env.example"; set +a
  echo "[scale_study] WARNING: using PROTOCOL.env.example — copy it to PROTOCOL.env" >&2
fi

ROOT="${ROOT:-${_ROOT_DEFAULT}}"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/scale300k}"
RESULTS_DIR="${RESULTS_DIR:-results/scale_study}"
LOG_DIR="${LOG_DIR:-logs/scale_study}"
RUNS_TSV="${RUNS_TSV:-${_SCALE_DIR}/runs.tsv}"
FROZEN_JSON="${FROZEN_JSON:-${RESULTS_DIR}/frozen.json}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${DATA_DIR}"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
need_cmd()  { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

py() { "${PYTHON}" "$@"; }

# Read a TSV row (skip comments / blank). Args: run_id → prints fields.
lookup_run() {
  local run_id="$1"
  awk -F'\t' -v id="${run_id}" '
    /^#/ || NF==0 { next }
    NR==1 && $1=="run_id" { next }
    $1==id { print; found=1; exit }
    END { if (!found) exit 1 }
  ' "${RUNS_TSV}"
}

manifest_for_n() {
  local n="$1"
  echo "${DATA_DIR}/manifest_train_${n}.parquet"
}

# Best ModelCheckpoint by lowest val_mismatch in filename, else last.ckpt.
best_checkpoint() {
  local run_id="$1"
  local base="${LOG_DIR}/${run_id}"
  [[ -d "${base}" ]] || die "No log dir for run ${run_id}: ${base}"
  # Prefer newest lightning version_* directory.
  local vdir
  vdir="$(ls -d "${base}"/version_* 2>/dev/null | sort -V | tail -1)" || true
  [[ -n "${vdir}" ]] || die "No version_* under ${base}"
  local ckpt_dir="${vdir}/checkpoints"
  [[ -d "${ckpt_dir}" ]] || die "No checkpoints in ${ckpt_dir}"

  # Parse val_mismatch= from filenames; pick minimum.
  local best=""
  local best_val="1e99"
  local f val
  for f in "${ckpt_dir}"/*.ckpt; do
    [[ -f "$f" ]] || continue
    if [[ "$(basename "$f")" == "last.ckpt" ]]; then
      continue
    fi
    val="$(basename "$f" | sed -n 's/.*val_mismatch=\([0-9.eE+-]*\).*/\1/p')"
    if [[ -n "${val}" ]]; then
      if py -c "import sys; sys.exit(0 if float('${val}') < float('${best_val}') else 1)"; then
        best_val="${val}"
        best="${f}"
      fi
    fi
  done
  if [[ -z "${best}" ]]; then
    [[ -f "${ckpt_dir}/last.ckpt" ]] || die "No usable checkpoint in ${ckpt_dir}"
    best="${ckpt_dir}/last.ckpt"
  fi
  echo "${best}"
}

residual_scale_from_frozen() {
  need_file "${FROZEN_JSON}"
  py -c "import json; print(json.load(open('${FROZEN_JSON}'))['residual_scale'])"
}

require_prediction_before_heldout() {
  local phase="$1"
  if [[ "${phase}" == "heldout" ]]; then
    need_file "${RESULTS_DIR}/prediction.json"
    log "Held-out gate OK: ${RESULTS_DIR}/prediction.json exists"
  fi
}
