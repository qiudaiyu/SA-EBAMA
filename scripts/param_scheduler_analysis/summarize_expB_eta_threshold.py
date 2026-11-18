import json
import re
from pathlib import Path


import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("outputs/param_schedule_analysis/expB_eta_threshold")
RESULT_ROOT = Path("results/param_schedule_analysis")
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def trace_value(item, key):
    """Compatible with direct fields, loss_dict, last_loss_dict, and losses."""
    if key in item:
        return item.get(key)
    for sub_key in ["loss_dict", "last_loss_dict", "losses"]:
        obj = item.get(sub_key)
        if isinstance(obj, dict) and key in obj:
            return obj.get(key)
    return None


def parse_eta(method):
    # B_eta030 -> 0.30, B_eta045 -> 0.45
    m = re.match(r"B_eta(\d+)$", str(method))
    if not m:
        return np.nan
    return int(m.group(1)) / 100.0


def bootstrap_ci(x, n_boot=2000, seed=1234):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        means.append(rng.choice(x, size=len(x), replace=True).mean())
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def collect():
    rows = []

    for p in OUTPUT_ROOT.rglob("*_guidance_trace.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            rows.append({
                "json_path": str(p),
                "bad_json": True,
                "error": str(e),
            })
            continue

        trace = data.get("trace", [])
        attempted = [x for x in trace if x.get("attempted", False)]
        updated = [x for x in trace if x.get("updated", False)]

        overlaps = [safe_float(trace_value(x, "overlap")) for x in attempted]
        overlaps = [x for x in overlaps if not np.isnan(x)]

        seps = [safe_float(trace_value(x, "sep")) for x in attempted]
        seps = [x for x in seps if not np.isnan(x)]

        totals = [safe_float(trace_value(x, "total")) for x in attempted]
        totals = [x for x in totals if not np.isnan(x)]

        step_sizes = [safe_float(trace_value(x, "step_size")) for x in attempted]
        step_sizes = [x for x in step_sizes if not np.isnan(x)]

        method = data.get("method")

        rows.append({
            "json_path": str(p),
            "bad_json": False,
            "method": method,
            "eta": parse_eta(method),
            "prompt_id": data.get("prompt_id"),
            "prompt": data.get("prompt"),
            "seed": data.get("seed"),
            "image_path": data.get("image_path"),
            "time_seconds": safe_float(data.get("time_seconds")),
            "peak_memory_mb": safe_float(data.get("peak_memory_mb")),
            "num_attempted_steps": safe_float(data.get("num_attempted_steps")),
            "num_updated_steps": safe_float(data.get("num_updated_steps")),
            "mean_overlap_attempted": float(np.mean(overlaps)) if overlaps else np.nan,
            "last_overlap_attempted": float(overlaps[-1]) if overlaps else np.nan,
            "min_overlap_attempted": float(np.min(overlaps)) if overlaps else np.nan,
            "mean_sep_attempted": float(np.mean(seps)) if seps else np.nan,
            "last_sep_attempted": float(seps[-1]) if seps else np.nan,
            "mean_total_loss_attempted": float(np.mean(totals)) if totals else np.nan,
            "mean_step_size_attempted": float(np.mean(step_sizes)) if step_sizes else np.nan,
            "num_trace_items": len(trace),
            "num_attempted_items_from_trace": len(attempted),
            "num_updated_items_from_trace": len(updated),
        })

    df = pd.DataFrame(rows)
    out_csv = RESULT_ROOT / "expB_eta_threshold_trace_metrics.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df


def summarize(df):
    ok = df[(df["bad_json"] == False) & df["method"].notna()].copy()

    summary = ok.groupby(["eta", "method"]).agg(
        n=("image_path", "count"),
        time_image=("time_seconds", "mean"),
        time_std=("time_seconds", "std"),
        update_steps=("num_updated_steps", "mean"),
        attempted_steps=("num_attempted_steps", "mean"),
        mean_overlap=("mean_overlap_attempted", "mean"),
        last_overlap=("last_overlap_attempted", "mean"),
        min_overlap=("min_overlap_attempted", "mean"),
        mean_sep=("mean_sep_attempted", "mean"),
        last_sep=("last_sep_attempted", "mean"),
        mean_total_loss=("mean_total_loss_attempted", "mean"),
        mean_step_size=("mean_step_size_attempted", "mean"),
        peak_memory=("peak_memory_mb", "mean"),
    ).reset_index().sort_values("eta")

    out_csv = RESULT_ROOT / "expB_eta_threshold_trace_summary.csv"
    summary.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    paper = summary[[
        "eta",
        "n",
        "update_steps",
        "attempted_steps",
        "time_image",
        "mean_overlap",
        "last_overlap",
        "mean_sep",
        "peak_memory",
    ]].copy()

    paper_out = RESULT_ROOT / "expB_eta_threshold_for_paper.csv"
    paper.to_csv(paper_out, index=False)
    print("Saved:", paper_out)

    return summary


def paired_vs_eta045(df):
    ok = df[(df["bad_json"] == False) & df["method"].notna()].copy()

    base_method = "B_eta045"
    if base_method not in set(ok["method"]):
        print("[WARN] Base method B_eta045 not found. Skip paired summary.")
        return None

    base = ok[ok["method"] == base_method].copy()

    metrics = [
        "time_seconds",
        "num_updated_steps",
        "num_attempted_steps",
        "mean_overlap_attempted",
        "last_overlap_attempted",
        "mean_sep_attempted",
    ]

    rows = []

    for method in sorted(ok["method"].unique()):
        if method == base_method:
            continue

        cur = ok[ok["method"] == method].copy()
        merged = cur.merge(
            base,
            on=["prompt_id", "seed"],
            suffixes=("_method", "_base")
        )

        eta = parse_eta(method)

        for metric in metrics:
            a = merged[f"{metric}_method"]
            b = merged[f"{metric}_base"]

            # All listed metrics are lower-is-better for this diagnostic analysis.
            # delta = base - method; positive means current method is lower than eta=0.45.
            delta = b - a
            delta = delta.dropna()

            ci_l, ci_h = bootstrap_ci(delta.values)

            rows.append({
                "base": base_method,
                "method": method,
                "eta": eta,
                "metric": metric,
                "n": int(delta.shape[0]),
                "mean_delta_positive_means_lower_than_eta045": float(delta.mean()) if len(delta) else np.nan,
                "ci_low": ci_l,
                "ci_high": ci_h,
                "win_rate": float((delta > 0).mean()) if len(delta) else np.nan,
            })

    out = pd.DataFrame(rows).sort_values(["eta", "metric"])
    out_csv = RESULT_ROOT / "expB_eta_threshold_paired_vs_eta045.csv"
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return out


def main():
    df = collect()
    summary = summarize(df)
    paired = paired_vs_eta045(df)

    print("\n=== Summary for paper ===")
    cols = ["eta", "n", "update_steps", "attempted_steps", "time_image", "mean_overlap", "last_overlap", "mean_sep"]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
