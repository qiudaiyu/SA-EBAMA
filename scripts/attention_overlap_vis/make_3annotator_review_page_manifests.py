import argparse
from pathlib import Path
import pandas as pd


CASE_ORDER = {
    "combined_repair": 0,
    "object_entanglement_repair": 1,
    "weak_object_repair": 2,
    "binding_repair": 3,
    "neutral": 4,
    "failure_or_tradeoff": 5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--page_size", type=int, default=3)
    ap.add_argument("--max_per_type", type=int, default=18)
    args = ap.parse_args()

    df = pd.read_csv(args.candidate_csv)

    df["case_order"] = df["case_type"].map(CASE_ORDER).fillna(999).astype(int)
    df = df.sort_values(["case_order", "selection_score"], ascending=[True, False])

    parts = []
    for t, g in df.groupby("case_type", sort=False):
        if t == "failure_or_tradeoff":
            parts.append(g.sort_values("selection_score", ascending=True).head(args.max_per_type))
        else:
            parts.append(g.sort_values("selection_score", ascending=False).head(args.max_per_type))

    df = pd.concat(parts, ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_id = 1
    for i in range(0, len(df), args.page_size):
        sub = df.iloc[i:i + args.page_size].copy()
        fig = f"review_page_{page_id:03d}"

        out = pd.DataFrame({
            "figure": fig,
            "row_order": range(1, len(sub) + 1),
            "prompt_id": sub["prompt_id"].values,
            "prompt": sub["prompt"].values,
            "seed": sub["seed"].values,
            "case_type": sub["case_type"].values,
            "selection_score": sub["selection_score"].values,
        })

        out.to_csv(out_dir / f"{fig}.csv", index=False)
        page_id += 1

    print(f"Saved {page_id - 1} review manifests to {out_dir}")


if __name__ == "__main__":
    main()
