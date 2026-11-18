#!/usr/bin/env bash
set -euo pipefail

#   bash scripts/param_scheduler_analysis/run_param_schedule_analysis_all.sh all
#   bash scripts/param_scheduler_analysis/run_param_schedule_analysis_all.sh expB
#   bash scripts/param_scheduler_analysis/run_param_schedule_analysis_all.sh summarize

MODE="${1:-all}"

MODEL_PATH="${MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
FORMAL_CSV="${FORMAL_CSV:-dataset/formal_prompts_main_400.csv}"

RESULT_ROOT="results/param_schedule_analysis"
OUTPUT_ROOT="outputs/param_schedule_analysis"

CALIB60="${RESULT_ROOT}/calib60_prompts.csv"
CALIB20="${RESULT_ROOT}/calib20_prompts.csv"
CALIB60_TSV="${RESULT_ROOT}/calib60_prompts.tsv"
CALIB20_TSV="${RESULT_ROOT}/calib20_prompts.tsv"

SEEDS=(1 42 1234 70690)

mkdir -p "${RESULT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

prepare_prompts () {
  echo "============================================================"
  echo "[Prepare] calibration prompt files"
  echo "============================================================"

  if [[ ! -f "${CALIB60}" ]]; then
    if [[ ! -f "${FORMAL_CSV}" ]]; then
      echo "[ERROR] Cannot find ${CALIB60} or ${FORMAL_CSV}"
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
print("Saved", out_csv)
print(calib["dataset"].value_counts())
print("Total", len(calib))
PY
  else
    echo "[OK] Found existing ${CALIB60}"
  fi

  python - "${CALIB60}" "${CALIB60_TSV}" "${CALIB20}" "${CALIB20_TSV}" <<'PY'
import pandas as pd
from pathlib import Path
import sys

calib60 = Path(sys.argv[1])
calib20 = Path(sys.argv[3])
tsv60 = Path(sys.argv[2])
tsv20 = Path(sys.argv[4])

df = pd.read_csv(calib60)

calib20_df = df.groupby("dataset", sort=True).head(5).reset_index(drop=True)
calib20_df.to_csv(calib20, index=False)

df[["prompt_id", "prompt"]].to_csv(tsv60, sep="\t", index=False, header=False)
calib20_df[["prompt_id", "prompt"]].to_csv(tsv20, sep="\t", index=False, header=False)

print("calib60 prompts:", len(df), "->", tsv60)
print("calib20 prompts:", len(calib20_df), "->", tsv20)
print(calib20_df["dataset"].value_counts())
PY
}

is_completed () {
  local EXP_DIR="$1"
  local METHOD="$2"
  local PROMPT_ID="$3"
  local SEED="$4"

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

run_one () {
  local EXP_DIR="$1"
  local METHOD="$2"
  local PROMPT_ID="$3"
  local PROMPT="$4"
  local SEED="$5"

  local USE_SEP="$6"
  local GAMMA="$7"
  local PCT="$8"
  local TEMP="$9"

  local USE_ADP="${10}"
  local ADP_MODE="${11}"
  local OVERLAP_THR="${12}"
  local DYNAMIC="${13}"

  if is_completed "${EXP_DIR}" "${METHOD}" "${PROMPT_ID}" "${SEED}"; then
    echo "[SKIP] ${METHOD} | ${PROMPT_ID} | seed=${SEED}"
    return 0
  fi

  local SEP_FLAG="--use_separation"
  if [[ "${USE_SEP}" == "no" ]]; then
    SEP_FLAG="--no-use_separation"
  fi

  local ADP_FLAG="--use_adaptive_guidance"
  if [[ "${USE_ADP}" == "no" ]]; then
    ADP_FLAG="--no-use_adaptive_guidance"
  fi

  local DYN_FLAG="--adaptive_dynamic_step"
  if [[ "${DYNAMIC}" == "no" ]]; then
    DYN_FLAG="--no-adaptive_dynamic_step"
  fi

  echo "[RUN] ${METHOD} | ${PROMPT_ID} | seed=${SEED}"

  python -u inference.py \
    --prompt "${PROMPT}" \
    --prompt_id "${PROMPT_ID}" \
    --method_name "${METHOD}" \
    --seed "${SEED}" \
    --output_directory "${EXP_DIR}/${METHOD}" \
    --model_path "${MODEL_PATH}" \
    --step_size 20 \
    --attn_res 256 \
    --gpu "${GPU:-0}" \
    --dist cos \
    --lambda_ours 0.5 \
    ${SEP_FLAG} \
    --gamma_sep "${GAMMA}" \
    --sep_tau percentile \
    --sep_percentile "${PCT}" \
    --sep_temp "${TEMP}" \
    ${ADP_FLAG} \
    --adaptive_mode "${ADP_MODE}" \
    --max_guidance_step 25 \
    --adaptive_overlap_threshold "${OVERLAP_THR}" \
    ${DYN_FLAG} \
    --log_guidance \
    --force_ddim
}

run_expA_gamma () {
  local EXP_DIR="${OUTPUT_ROOT}/expA_gamma"
  mkdir -p "${EXP_DIR}"

  echo "============================================================"
  echo "[Experiment A1] gamma_sep analysis"
  echo "============================================================"

  # method gamma use_separation
  local SETTINGS=(
    "A_g000 0.00 no"
    "A_g0025 0.025 yes"
    "A_g005 0.05 yes"
    "A_g010 0.10 yes"
    "A_g020 0.20 yes"
  )

  for item in "${SETTINGS[@]}"; do
    read -r METHOD GAMMA USE_SEP <<< "${item}"

    echo "------------------------------------------------------------"
    echo "Running / Resuming ${METHOD}, gamma=${GAMMA}"
    echo "------------------------------------------------------------"

    while IFS=$'\t' read -r PROMPT_ID PROMPT; do
      for SEED in "${SEEDS[@]}"; do
        run_one \
          "${EXP_DIR}" "${METHOD}" "${PROMPT_ID}" "${PROMPT}" "${SEED}" \
          "${USE_SEP}" "${GAMMA}" "0.70" "0.15" \
          "no" "fixed25" "999" "no"
      done
    done < "${CALIB60_TSV}"
  done
}

run_expA_mask () {
  local EXP_DIR="${OUTPUT_ROOT}/expA_mask"
  mkdir -p "${EXP_DIR}"

  echo "============================================================"
  echo "[Experiment A2] sep_percentile and temperature analysis"
  echo "============================================================"

  for PCT in 0.60 0.70 0.80; do
    for TEMP in 0.10 0.15 0.20; do
      METHOD="A_pct${PCT}_temp${TEMP}"
      METHOD=${METHOD//./}

      echo "------------------------------------------------------------"
      echo "Running / Resuming ${METHOD}, sep_percentile=${PCT}, sep_temp=${TEMP}"
      echo "------------------------------------------------------------"

      while IFS=$'\t' read -r PROMPT_ID PROMPT; do
        for SEED in "${SEEDS[@]}"; do
          run_one \
            "${EXP_DIR}" "${METHOD}" "${PROMPT_ID}" "${PROMPT}" "${SEED}" \
            "yes" "0.05" "${PCT}" "${TEMP}" \
            "no" "fixed25" "999" "no"
        done
      done < "${CALIB20_TSV}"
    done
  done
}

run_expB_schedule () {
  local EXP_DIR="${OUTPUT_ROOT}/expB_schedule"
  mkdir -p "${EXP_DIR}"

  echo "============================================================"
  echo "[Experiment B] timestep scheduling analysis"
  echo "============================================================"

  # method adaptive_mode overlap_threshold dynamic
  local SETTINGS=(
    "B_fixed25 fixed25 999 no"
    "B_early10 early10 999 no"
    "B_uniform10 uniform10 999 no"
    "B_sparse10 sparse10 999 no"
    "B_sparse10_trigger sparse10 0.45 no"
    "B_final sparse10 0.45 yes"
  )

  for item in "${SETTINGS[@]}"; do
    read -r METHOD ADP_MODE OVERLAP_THR DYNAMIC <<< "${item}"

    echo "------------------------------------------------------------"
    echo "Running / Resuming ${METHOD}, mode=${ADP_MODE}, threshold=${OVERLAP_THR}, dynamic=${DYNAMIC}"
    echo "------------------------------------------------------------"

    while IFS=$'\t' read -r PROMPT_ID PROMPT; do
      for SEED in "${SEEDS[@]}"; do
        run_one \
          "${EXP_DIR}" "${METHOD}" "${PROMPT_ID}" "${PROMPT}" "${SEED}" \
          "yes" "0.05" "0.70" "0.15" \
          "yes" "${ADP_MODE}" "${OVERLAP_THR}" "${DYNAMIC}"
      done
    done < "${CALIB60_TSV}"
  done
}

run_summarize () {
  echo "============================================================"
  echo "[Summarize] collect metrics and draw plots"
  echo "============================================================"

  python scripts/param_scheduler_analysis/summarize_param_schedule_analysis_all.py
}

prepare_prompts

case "${MODE}" in
  all)
    run_expA_gamma
    run_expA_mask
    run_expB_schedule
    run_summarize
    ;;
  expA)
    run_expA_gamma
    run_expA_mask
    ;;
  expA_gamma)
    run_expA_gamma
    ;;
  expA_mask)
    run_expA_mask
    ;;
  expB)
    run_expB_schedule
    ;;
  summarize)
    run_summarize
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Available modes: all | expA | expA_gamma | expA_mask | expB | summarize"
    exit 1
    ;;
esac

echo "All requested tasks finished."
