import argparse
from pathlib import Path

import pandas as pd


# v2: create a paper-oriented failure-case manifest and analysis table from
# the three-annotator qualitative candidate CSV.


def infer_failure_type(row):
    types = []
    if float(row.get("fusion_delta", 0)) < 0:
        types.append("Fusion / boundary ambiguity worsened")
    if float(row.get("missing_delta", 0)) < 0:
        types.append("Missing or weak-object expression worsened")
    if float(row.get("binding_error_delta", 0)) < 0:
        types.append("Attribute binding worsened")
    if float(row.get("correct_delta", 0)) < 0:
        types.append("Object correctness worsened")
    if not types:
        types.append("Qualitative trade-off")
    return "; ".join(types)


def infer_reason(row):
    p = str(row.get("prompt", "")).lower()
    ftype = infer_failure_type(row)

    reasons = []
    if any(x in p for x in ["left of", "right of", "top of", "bottom of", "above", "below", "next to"]):
        reasons.append("the prompt contains an explicit spatial relation, while SA-EBAMA does not impose geometric layout constraints")
    if "and" in p or "," in p:
        reasons.append("multiple objects or modifiers may compete for overlapping cross-attention responses")
    if "Fusion" in ftype or "boundary" in ftype:
        reasons.append("attention-level separation does not directly optimize pixel-level instance boundaries")
    if "Missing" in ftype or "weak-object" in ftype:
        reasons.append("the weak or small object can still be suppressed by the base diffusion prior")
    if "binding" in ftype.lower():
        reasons.append("object-token extraction and attention aggregation can be noisy for complex phrases")
    if not reasons:
        reasons.append("the case reflects the remaining gap between attention-level guidance and final image perception")

    return "; ".join(dict.fromkeys(reasons))


def infer_paper_sentence(row):
    ftype = infer_failure_type(row)
    if "Missing" in ftype and "Fusion" in ftype:
        return "SA-EBAMA reduces attention-level overlap on average, but this case shows that weak-object preservation and visible object separation can still fail simultaneously."
    if "Missing" in ftype:
        return "This case shows that object-level separation guidance cannot guarantee the visibility of every weak object."
    if "Fusion" in ftype or "boundary" in ftype:
        return "This case shows that lower attention overlap should not be interpreted as a guaranteed reduction of human-visible fusion."
    if "binding" in ftype.lower():
        return "This case indicates that binding errors are reduced statistically but not fully eliminated."
    return "This case is included to show a qualitative trade-off rather than a uniform visual improvement."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_csv", required=True)
    ap.add_argument("--out_manifest_csv", required=True)
    ap.add_argument("--out_analysis_csv", required=True)
    ap.add_argument("--out_analysis_md", default=None)
    ap.add_argument("--figure", default="figure_failure_cases")
    ap.add_argument("--max_cases", type=int, default=4)
    ap.add_argument("--include_selected_paper_failure", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.candidate_csv)

    required = ["prompt_id", "prompt", "seed", "case_type", "selection_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Existing columns: {df.columns.tolist()}")

    # Primary failure pool: cases explicitly tagged as failure/trade-off or any negative paired delta.
    delta_cols = [c for c in ["fusion_delta", "missing_delta", "binding_error_delta", "correct_delta"] if c in df.columns]
    if delta_cols:
        neg_mask = (df[delta_cols].fillna(0).astype(float) < 0).any(axis=1)
    else:
        neg_mask = False

    pool = df[df["case_type"].eq("failure_or_tradeoff") | neg_mask].copy()
    if pool.empty:
        raise ValueError("No failure/trade-off cases found. Check case_type or delta columns.")

    # Put manually selected paper failure first if requested, then the most severe negative-score cases.
    pool["paper_failure_rank"] = pool.get("paper_use", "").astype(str).eq("figure2_failure").map({True: 0, False: 1})
    pool = pool.sort_values(["paper_failure_rank", "selection_score"], ascending=[True, True]).head(args.max_cases)

    manifest = pd.DataFrame({
        "figure": args.figure,
        "row_order": range(1, len(pool) + 1),
        "prompt_id": pool["prompt_id"].astype(str).values,
        "prompt": pool["prompt"].astype(str).values,
        "seed": pool["seed"].astype(int).values,
        "case_type": pool["case_type"].astype(str).values,
        "selection_score": pool["selection_score"].values,
        "failure_type": [infer_failure_type(r) for _, r in pool.iterrows()],
        "failure_reason": [infer_reason(r) for _, r in pool.iterrows()],
    })

    analysis = manifest[[
        "row_order", "prompt_id", "prompt", "seed", "failure_type", "failure_reason"
    ]].copy()
    analysis["paper_interpretation"] = [infer_paper_sentence(r) for _, r in pool.iterrows()]

    Path(args.out_manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_analysis_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_manifest_csv, index=False)
    analysis.to_csv(args.out_analysis_csv, index=False)

    if args.out_analysis_md:
        Path(args.out_analysis_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_analysis_md, "w", encoding="utf-8") as f:
            f.write("# Failure-case analysis table\n\n")
            f.write(analysis.to_markdown(index=False))
            f.write("\n")

    print("Saved manifest:", args.out_manifest_csv)
    print("Saved analysis CSV:", args.out_analysis_csv)
    if args.out_analysis_md:
        print("Saved analysis MD:", args.out_analysis_md)
    print(analysis.to_string(index=False))


if __name__ == "__main__":
    main()
