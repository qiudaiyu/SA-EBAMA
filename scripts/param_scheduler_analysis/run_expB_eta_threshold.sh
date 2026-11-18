#!/usr/bin/env bash
set -euo pipefail

# bash scripts/param_scheduler_analysis/run_expB_eta_threshold.sh quick
# bash scripts/param_scheduler_analysis/run_expB_eta_threshold.sh full

MODE="${1:-quick}"

MODEL_PATH="${MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
FORMAL_CSV="${FORMAL_CSV:-dataset/formal_prompts_main_400.csv}"

RESULT_ROOT="results/param_schedule_analysis"
OUTPUT_ROOT="outputs/param_schedule_analysis"

CALIB60="${RESULT_ROOT}/calib60_prompts.csv"
CALIB60_TSV="${RESULT_ROOT}/calib60_prompts.tsv"

EXP_DIR="${OUTPUT_ROOT}/expB_eta_threshold"

GPU="${GPU:-0}"

mkdir -p "${RESULT_ROOT}"
mkdir -p "${EXP_DIR}"

# Prepare calibration prompt subset: 15 prompts per dataset, 60 total
prepare_prompts () {
  echo "============================================================"
  echo "[Prepare] calibration prompt files"
  echo "============================================================"

  if [[ ! -f "${CALIB60}" ]]; then
    if [[ ! -f "${FORMAL_CSV}" ]]; then
      echo "[ERROR] Cannot find ${FORMAL_CSV}"
      echo "Set FORMAL_CSV to the path of the benchmark CSV."
      exit 1
    fi

    python - "${FORMAL_CSV}" "${CALIB60}" <<'PY'
import pandas as pd
from pathlib import Path
import sys

formal_csv, out_csv = map(Path, sys.argv[1:])
out_csv.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(formal_csv)

parts = []
for dataset, g in df.groupby("dataset", sort=True):
    parts.append(g.sort_values("prompt_id").head(15))

calib = pd.concat(parts, ignore_index=True)
calib.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print(calib["dataset"].value_counts())
print("Total prompts:", len(calib))
PY
  else
    echo "[OK] Found existing ${CALIB60}"
  fi

  python - "${CALIB60}" "${CALIB60_TSV}" <<'PY'
import pandas as pd
from pathlib import Path
import sys

calib60 = Path(sys.argv[1])
tsv60 = Path(sys.argv[2])

df = pd.read_csv(calib60)
df[["prompt_id", "prompt"]].to_csv(tsv60, sep="\t", index=False, header=False)

print("Saved:", tsv60)
print("Total prompts:", len(df))
PY
}

# Select seeds
if [[ "${MODE}" == "quick" ]]; then
  SEEDS=(1)
elif [[ "${MODE}" == "full" ]]; then
  SEEDS=(1 42 1234 70690)
else
  echo "[ERROR] Unknown mode: ${MODE}"
  echo "Available modes: quick | full"
  exit 1
fi

# eta thresholds to test
ETA_VALUES=(0.30 0.35 0.45 0.55 0.65)

# Completion check
is_completed () {
  local METHOD="$1"
  local PROMPT_ID="$2"
  local SEED="$3"

  local METHOD_DIR="${EXP_DIR}/${METHOD}"

  if [[ ! -d "${METHOD_DIR}" ]]; then
    return 1
  fi

  local TRACE_PATH
  local IMG_PATH

  TRACE_PATH=$(find "${METHOD_DIR}" -maxdepth 2 -type f \
    -path "*/${PROMPT_ID}__*/seed${SEED}_alpha20_guidance_trace.json" \
    -print -quit 2>/dev/null || true)

  IMG_PATH=$(find "${METHOD_DIR}" -maxdepth 2 -type f \
    -path "*/${PROMPT_ID}__*/seed${SEED}_alpha20.png" \
    -print -quit 2>/dev/null || true)

  if [[ -z "${TRACE_PATH}" || -z "${IMG_PATH}" ]]; then
    return 1
  fi

  python - "${TRACE_PATH}" "${SEED}" <<'PY'
import json
import sys

trace_path = sys.argv[1]
seed = int(sys.argv[2])

try:
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ok = (
        data.get("seed") == seed
        and isinstance(data.get("trace"), list)
        and len(data.get("trace")) > 0
    )
except Exception:
    ok = False

sys.exit(0 if ok else 1)
PY
}

# Run one case
run_one () {
  local METHOD="$1"
  local ETA="$2"
  local PROMPT_ID="$3"
  local PROMPT="$4"
  local SEED="$5"

  if is_completed "${METHOD}" "${PROMPT_ID}" "${SEED}"; then
    echo "[SKIP] ${METHOD} | ${PROMPT_ID} | seed=${SEED}"
    return 0
  fi

  echo "[RUN] ${METHOD} | eta=${ETA} | ${PROMPT_ID} | seed=${SEED}"

  python -u inference.py \
    --prompt "${PROMPT}" \
    --prompt_id "${PROMPT_ID}" \
    --method_name "${METHOD}" \
    --seed "${SEED}" \
    --output_directory "${EXP_DIR}/${METHOD}" \
    --model_path "${MODEL_PATH}" \
    --step_size 20 \
    --attn_res 256 \
    --gpu "${GPU}" \
    --dist cos \
    --lambda_ours 0.5 \
    --use_separation \
    --gamma_sep 0.05 \
    --sep_tau percentile \
    --sep_percentile 0.70 \
    --sep_temp 0.15 \
    --use_adaptive_guidance \
    --adaptive_mode sparse10 \
    --max_guidance_step 25 \
    --adaptive_overlap_threshold "${ETA}" \
    --adaptive_dynamic_step \
    --log_guidance \
    --force_ddim
}

# Main
prepare_prompts

echo "============================================================"
echo "[Experiment] Eta threshold sensitivity"
echo "Mode: ${MODE}"
echo "Seeds: ${SEEDS[*]}"
echo "Eta values: ${ETA_VALUES[*]}"
echo "Output: ${EXP_DIR}"
echo "============================================================"

for ETA in "${ETA_VALUES[@]}"; do
  METHOD="B_eta${ETA}"
  METHOD="${METHOD//./}"

  echo "------------------------------------------------------------"
  echo "Running / Resuming ${METHOD}, eta=${ETA}"
  echo "------------------------------------------------------------"

  while IFS=$'\t' read -r PROMPT_ID PROMPT; do
    for SEED in "${SEEDS[@]}"; do
      run_one "${METHOD}" "${ETA}" "${PROMPT_ID}" "${PROMPT}" "${SEED}"
    done
  done < "${CALIB60_TSV}"
done

echo "Eta threshold sensitivity experiment finished."
