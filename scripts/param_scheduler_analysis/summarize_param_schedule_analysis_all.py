

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("outputs/param_schedule_analysis")
RESULT_ROOT = Path("results/param_schedule_analysis")
PLOT_DIR = RESULT_ROOT / "plots"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def trace_value(item, key):
    """Read a trace field from the current or earlier JSON schema."""
    if key in item:
        return item.get(key)
    for sub_key in ["loss_dict", "last_loss_dict", "losses"]:
        obj = item.get(sub_key)
        if isinstance(obj, dict) and key in obj:
            return obj.get(key)
    return None


def parse_experiment_from_path(path: Path):
    parts = path.parts
    if "expA_gamma" in parts:
        return "expA_gamma"
    if "expA_mask" in parts:
        return "expA_mask"
    if "expB_schedule" in parts:
        return "expB_schedule"
    return "unknown"


def collect_trace_metrics():
    rows = []

    for p in OUTPUT_ROOT.rglob("*_guidance_trace.json"):
        experiment = parse_experiment_from_path(p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            rows.append({
                "json_path": str(p),
                "experiment": experiment,
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

        rows.append({
            "json_path": str(p),
            "experiment": experiment,
            "bad_json": False,
            "method": data.get("method"),
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
    out_csv = RESULT_ROOT / "trace_metrics_param_schedule.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df


def summarize(df):
    ok = df[(df.get("bad_json") == False) & df["method"].notna()].copy()

    summary = ok.groupby(["experiment", "method"]).agg(
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
    ).reset_index()

    all_path = RESULT_ROOT / "all_method_trace_summary.csv"
    summary.to_csv(all_path, index=False)
    print("Saved:", all_path)

    for exp in ["expA_gamma", "expA_mask", "expB_schedule"]:
        sub = summary[summary["experiment"] == exp].copy()
        if not sub.empty:
            out_path = RESULT_ROOT / f"{exp}_trace_summary.csv"
            sub.to_csv(out_path, index=False)
            print("Saved:", out_path)

    return summary


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


def paired_summary(df, experiment, base_method, out_name):
    ok = df[
        (df.get("bad_json") == False)
        & (df["experiment"] == experiment)
        & df["method"].notna()
    ].copy()

    if ok.empty or base_method not in set(ok["method"]):
        print(f"[SKIP paired] {experiment}: base {base_method} not found.")
        return None

    metrics = [
        "time_seconds",
        "num_updated_steps",
        "num_attempted_steps",
        "mean_overlap_attempted",
        "last_overlap_attempted",
        "mean_sep_attempted",
    ]


    directions = {
        "time_seconds": "low",
        "num_updated_steps": "low",
        "num_attempted_steps": "low",
        "mean_overlap_attempted": "low",
        "last_overlap_attempted": "low",
        "mean_sep_attempted": "low",
    }

    base = ok[ok["method"] == base_method].copy()
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

        for metric in metrics:
            a = merged[f"{metric}_method"]
            b = merged[f"{metric}_base"]

            if directions[metric] == "low":
                delta = b - a
            else:
                delta = a - b

            ci_l, ci_h = bootstrap_ci(delta.dropna().values)

            rows.append({
                "experiment": experiment,
                "base": base_method,
                "method": method,
                "metric": metric,
                "n": int(delta.dropna().shape[0]),
                "mean_delta_positive_is_better": float(delta.mean()) if len(delta) else np.nan,
                "ci_low": ci_l,
                "ci_high": ci_h,
                "win_rate": float((delta > 0).mean()) if len(delta) else np.nan,
            })

    out = pd.DataFrame(rows)
    out_path = RESULT_ROOT / out_name
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return out


def parse_gamma(method):
    mapping = {
        "A_g000": 0.0,
        "A_g0025": 0.025,
        "A_g005": 0.05,
        "A_g010": 0.10,
        "A_g020": 0.20,
    }
    return mapping.get(method, np.nan)


def parse_pct_temp(method):
    # A_pct070_temp015 -> 0.70, 0.15
    m = re.match(r"A_pct(\d+)_temp(\d+)", str(method))
    if not m:
        return np.nan, np.nan
    pct = int(m.group(1)) / 100
    temp = int(m.group(2)) / 100
    return pct, temp


def plot_gamma(summary):
    sub = summary[summary["experiment"] == "expA_gamma"].copy()
    if sub.empty:
        return
    sub["gamma"] = sub["method"].map(parse_gamma)
    sub = sub.dropna(subset=["gamma"]).sort_values("gamma")

    if sub.empty:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(sub["gamma"], sub["mean_overlap"], marker="o")
    plt.xlabel("gamma_sep")
    plt.ylabel("Mean overlap during attempted guidance")
    plt.title("Effect of separation strength on attention overlap")
    plt.tight_layout()
    path = PLOT_DIR / "expA_gamma_mean_overlap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(6, 4))
    plt.plot(sub["gamma"], sub["time_image"], marker="o")
    plt.xlabel("gamma_sep")
    plt.ylabel("Time per image (s)")
    plt.title("Runtime under different separation strengths")
    plt.tight_layout()
    path = PLOT_DIR / "expA_gamma_time_image.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def plot_mask(summary):
    sub = summary[summary["experiment"] == "expA_mask"].copy()
    if sub.empty:
        return

    parsed = sub["method"].apply(parse_pct_temp)
    sub["sep_percentile"] = [x[0] for x in parsed]
    sub["sep_temp"] = [x[1] for x in parsed]
    sub = sub.dropna(subset=["sep_percentile", "sep_temp"])

    if sub.empty:
        return

    pivot = sub.pivot(index="sep_percentile", columns="sep_temp", values="mean_overlap")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
    plt.xlabel("temperature")
    plt.ylabel("sep percentile")
    plt.title("Mean overlap under different soft-mask parameters")
    plt.colorbar(label="Mean overlap")
    plt.tight_layout()
    path = PLOT_DIR / "expA_mask_mean_overlap_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)

    pivot_time = sub.pivot(index="sep_percentile", columns="sep_temp", values="time_image")
    pivot_time = pivot_time.sort_index().sort_index(axis=1)

    plt.figure(figsize=(6, 4))
    plt.imshow(pivot_time.values, aspect="auto")
    plt.xticks(range(len(pivot_time.columns)), [f"{x:.2f}" for x in pivot_time.columns])
    plt.yticks(range(len(pivot_time.index)), [f"{x:.2f}" for x in pivot_time.index])
    plt.xlabel("temperature")
    plt.ylabel("sep percentile")
    plt.title("Time per image under different soft-mask parameters")
    plt.colorbar(label="Time per image (s)")
    plt.tight_layout()
    path = PLOT_DIR / "expA_mask_time_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def plot_schedule(summary):
    sub = summary[summary["experiment"] == "expB_schedule"].copy()
    if sub.empty:
        return

    order = [
        "B_fixed25",
        "B_early10",
        "B_uniform10",
        "B_sparse10",
        "B_sparse10_trigger",
        "B_final",
    ]
    sub["order"] = sub["method"].map({m: i for i, m in enumerate(order)})
    sub = sub.dropna(subset=["order"]).sort_values("order")

    if sub.empty:
        return

    labels = sub["method"].tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(labels, sub["update_steps"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average updated steps")
    plt.title("Updated steps under different timestep schedules")
    plt.tight_layout()
    path = PLOT_DIR / "expB_update_steps.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, sub["time_image"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Time per image (s)")
    plt.title("Runtime under different timestep schedules")
    plt.tight_layout()
    path = PLOT_DIR / "expB_time_image.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, sub["mean_overlap"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean overlap during attempted guidance")
    plt.title("Attention overlap under different timestep schedules")
    plt.tight_layout()
    path = PLOT_DIR / "expB_mean_overlap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def main():
    df = collect_trace_metrics()
    summary = summarize(df)

    paired_summary(df, "expA_gamma", "A_g000", "expA_gamma_paired_vs_g000.csv")
    paired_summary(df, "expA_mask", "A_pct070_temp015", "expA_mask_paired_vs_final.csv")
    paired_summary(df, "expB_schedule", "B_fixed25", "expB_schedule_paired_vs_fixed25.csv")

    plot_gamma(summary)
    plot_mask(summary)
    plot_schedule(summary)

    print("\nDone.")
    print("Main outputs are in:", RESULT_ROOT)
    print("Plots are in:", PLOT_DIR)


if __name__ == "__main__":
    main()
