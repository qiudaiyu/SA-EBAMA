

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_vis_dir(out_root: Path, method: str, prompt_id: str, seed: int):
    """
    Expected structure:
    out_root / method / prompt_id__prompt / seed{seed}_alpha20_overlap_vis / avg_overlap_heatmap.npy

    This function is intentionally tolerant because prompt strings may contain spaces.
    """
    method_root = out_root / method
    if not method_root.exists():
        return None

    pattern = f"**/seed{seed}_alpha20_overlap_vis/avg_overlap_heatmap.npy"
    matches = list(method_root.glob(pattern))

    # Prefer path containing prompt_id
    matches = [p for p in matches if prompt_id in str(p)]

    if not matches:
        return None

    # If multiple matches exist, use the shortest path as the most likely exact one.
    matches = sorted(matches, key=lambda p: len(str(p)))
    return matches[0].parent


def read_avg_oor(vis_dir: Path):
    """
    Priority:
    1. meta.json: avg_oor / mean_oor / oor / avg_overlap / mean_overlap
    2. step_overlap_records.npz: oor_values / overlap_values / overlaps
    3. avg_overlap_heatmap.npy mean as fallback
    """
    meta_file = vis_dir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            for key in ["avg_oor", "mean_oor", "oor", "avg_overlap", "mean_overlap"]:
                if key in meta:
                    return float(meta[key]), "meta.json:" + key
        except Exception:
            pass

    records_file = vis_dir / "step_overlap_records.npz"
    if records_file.exists():
        try:
            records = np.load(records_file)
            for key in ["oor_values", "overlap_values", "overlaps"]:
                if key in records:
                    arr = np.asarray(records[key], dtype=float)
                    if arr.size > 0:
                        return float(np.nanmean(arr)), "step_overlap_records.npz:" + key
        except Exception:
            pass

    heatmap_file = vis_dir / "avg_overlap_heatmap.npy"
    if heatmap_file.exists():
        try:
            h = np.load(heatmap_file)
            return float(np.nanmean(h)), "avg_overlap_heatmap.npy:mean_fallback"
        except Exception:
            pass

    return None, "missing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--min_delta_oor", type=float, default=0.005)
    args = parser.parse_args()

    candidate_csv = Path(args.candidate_csv)
    out_root = Path(args.out_root)

    if not candidate_csv.exists():
        raise FileNotFoundError(f"candidate_csv not found: {candidate_csv}")
    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    df = pd.read_csv(candidate_csv)

    required = {"prompt_id", "prompt", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in candidate_csv: {missing}")

    rows = []

    for _, row in df.iterrows():
        dataset = row.get("dataset", "")
        prompt_id = str(row["prompt_id"]).strip()
        prompt = str(row["prompt"]).strip()
        seed = int(row["seed"])

        ebama_vis_dir = find_vis_dir(out_root, "ebama", prompt_id, seed)
        sa_vis_dir = find_vis_dir(out_root, "sa_ebama", prompt_id, seed)

        if ebama_vis_dir is None or sa_vis_dir is None:
            rows.append({
                "dataset": dataset,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": seed,
                "status": "missing_vis_dir",
                "ebama_oor": np.nan,
                "sa_oor": np.nan,
                "delta_oor": np.nan,
                "relative_drop": np.nan,
                "pass_delta": False,
                "ebama_vis_dir": "" if ebama_vis_dir is None else str(ebama_vis_dir),
                "sa_vis_dir": "" if sa_vis_dir is None else str(sa_vis_dir),
                "ebama_source": "missing",
                "sa_source": "missing",
            })
            continue

        ebama_oor, ebama_source = read_avg_oor(ebama_vis_dir)
        sa_oor, sa_source = read_avg_oor(sa_vis_dir)

        if ebama_oor is None or sa_oor is None:
            rows.append({
                "dataset": dataset,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": seed,
                "status": "missing_oor",
                "ebama_oor": np.nan if ebama_oor is None else ebama_oor,
                "sa_oor": np.nan if sa_oor is None else sa_oor,
                "delta_oor": np.nan,
                "relative_drop": np.nan,
                "pass_delta": False,
                "ebama_vis_dir": str(ebama_vis_dir),
                "sa_vis_dir": str(sa_vis_dir),
                "ebama_source": ebama_source,
                "sa_source": sa_source,
            })
            continue

        delta = float(ebama_oor - sa_oor)
        relative_drop = float(delta / ebama_oor) if ebama_oor > 0 else np.nan

        rows.append({
            "dataset": dataset,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "seed": seed,
            "status": "ok",
            "ebama_oor": ebama_oor,
            "sa_oor": sa_oor,
            "delta_oor": delta,
            "relative_drop": relative_drop,
            "pass_delta": bool(delta > args.min_delta_oor),
            "ebama_vis_dir": str(ebama_vis_dir),
            "sa_vis_dir": str(sa_vis_dir),
            "ebama_heatmap": str(ebama_vis_dir / "avg_overlap_heatmap.npy"),
            "sa_heatmap": str(sa_vis_dir / "avg_overlap_heatmap.npy"),
            "ebama_source": ebama_source,
            "sa_source": sa_source,
        })

    out = pd.DataFrame(rows)

    # ok rows first, then larger delta
    if "delta_oor" in out.columns:
        out["_status_order"] = out["status"].apply(lambda x: 0 if x == "ok" else 1)
        out = out.sort_values(["_status_order", "delta_oor"], ascending=[True, False])
        out = out.drop(columns=["_status_order"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print("\nStatus counts:")
    print(out["status"].value_counts(dropna=False).to_string())

    ok = out[out["status"].eq("ok")].copy()
    if not ok.empty:
        print("\nTop examples:")
        cols = ["dataset", "prompt_id", "prompt", "seed", "ebama_oor", "sa_oor", "delta_oor", "relative_drop", "pass_delta"]
        print(ok[cols].head(30).to_string(index=False))
    else:
        print("\nNo ok examples found. Please check out_root and generated heatmap paths.")


if __name__ == "__main__":
    main()
