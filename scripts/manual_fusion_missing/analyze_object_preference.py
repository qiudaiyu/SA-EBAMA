#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def norm_method(x):
    x = str(x).strip().lower()
    x = x.replace("-", "_")
    x = x.replace(" ", "_")
    if x in ["saebama", "sa_ebama", "sa_ebama_v2"]:
        return "sa_ebama"
    if x in ["ebama"]:
        return "ebama"
    return x


def norm_choice(x):
    if pd.isna(x):
        return "unclear"
    x = str(x).strip().lower()
    x = x.replace("-", "_")
    x = x.replace(" ", "_")
    if x in ["left", "left_better", "a", "a_better"]:
        return "left"
    if x in ["right", "right_better", "b", "b_better"]:
        return "right"
    if x in ["tie", "same", "equal", "both_good"]:
        return "tie"
    if x in ["unclear", "unknown", "cannot_judge", "nan", ""]:
        return "unclear"
    return x


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def binom_two_sided_p(k, n):
    """Two-sided exact binomial test under p=0.5, no scipy required."""
    if n <= 0:
        return float("nan")
    lo = min(k, n - k)
    prob = 0.0
    for i in range(lo + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2 * prob)


def fleiss_kappa(count_df, categories):
    """
    count_df: one row per item, columns are counts for categories.
    """
    mat = count_df[categories].to_numpy(dtype=float)
    N = mat.shape[0]
    if N == 0:
        return float("nan")
    n = mat.sum(axis=1)
    if not (n == n[0]).all():
        return float("nan")
    n = n[0]
    if n <= 1:
        return float("nan")
    P_i = ((mat ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = mat.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()
    if abs(1 - P_e) < 1e-12:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_csvs", nargs="+", required=True,
                        help="Annotator CSV files, e.g. annotator_A.csv annotator_B.csv annotator_C.csv")
    parser.add_argument("--pairs_csv", default=None,
                        help="Optional pairs CSV for checking pair coverage.")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory.")
    parser.add_argument("--make_figures", action="store_true",
                        help="Save PDF bar figures.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_votes = []

    required_cols = {
        "pair_id", "dataset", "prompt_id", "prompt", "seed",
        "left_method", "right_method", "object_preference"
    }

    for ann_path in args.ann_csvs:
        ann_path = Path(ann_path)
        df = pd.read_csv(ann_path)

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{ann_path} missing columns: {sorted(missing)}")

        if "annotator_id" not in df.columns:
            df["annotator_id"] = ann_path.stem

        for _, r in df.iterrows():
            left_method = norm_method(r["left_method"])
            right_method = norm_method(r["right_method"])
            choice = norm_choice(r["object_preference"])

            if choice == "left":
                pref = left_method
            elif choice == "right":
                pref = right_method
            elif choice == "tie":
                pref = "tie"
            elif choice == "unclear":
                pref = "unclear"
            else:
                pref = "unclear"

            all_votes.append({
                "pair_id": r["pair_id"],
                "dataset": r["dataset"],
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "seed": r["seed"],
                "annotator_id": r["annotator_id"],
                "left_method": left_method,
                "right_method": right_method,
                "raw_object_preference": r["object_preference"],
                "preference_method": pref,
            })

    votes = pd.DataFrame(all_votes)

    valid_prefs = ["sa_ebama", "ebama", "tie", "unclear"]
    votes["preference_method"] = votes["preference_method"].where(
        votes["preference_method"].isin(valid_prefs),
        "unclear"
    )

    votes.to_csv(out_dir / "human_preference_votes.csv", index=False, encoding="utf-8-sig")

    # Optional pair coverage check
    if args.pairs_csv:
        pairs = pd.read_csv(args.pairs_csv)
        expected_pairs = set(pairs["pair_id"].astype(str))
        observed_pairs = set(votes["pair_id"].astype(str))
        missing_pairs = sorted(expected_pairs - observed_pairs)
        extra_pairs = sorted(observed_pairs - expected_pairs)
        coverage = pd.DataFrame([{
            "n_pairs_expected": len(expected_pairs),
            "n_pairs_observed": len(observed_pairs),
            "n_missing_pairs": len(missing_pairs),
            "n_extra_pairs": len(extra_pairs),
            "missing_pairs": ";".join(missing_pairs[:20]),
            "extra_pairs": ";".join(extra_pairs[:20]),
        }])
        coverage.to_csv(out_dir / "human_preference_pair_coverage.csv", index=False, encoding="utf-8-sig")

    # Vote-level summary
    vote_counts = votes["preference_method"].value_counts().reindex(valid_prefs, fill_value=0)
    vote_summary_rows = []
    total_votes = int(vote_counts.sum())
    for pref in valid_prefs:
        k = int(vote_counts[pref])
        ci_low, ci_high = wilson_ci(k, total_votes)
        vote_summary_rows.append({
            "preference": pref,
            "count": k,
            "rate": k / total_votes if total_votes else float("nan"),
            "rate_percent": 100 * k / total_votes if total_votes else float("nan"),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })

    decisive_vote_n = int(vote_counts["sa_ebama"] + vote_counts["ebama"])
    decisive_vote_sa = int(vote_counts["sa_ebama"])
    decisive_vote_ebama = int(vote_counts["ebama"])
    p_value_vote = binom_two_sided_p(decisive_vote_sa, decisive_vote_n)

    vote_summary = pd.DataFrame(vote_summary_rows)
    vote_summary.to_csv(out_dir / "human_preference_vote_summary.csv", index=False, encoding="utf-8-sig")

    # Majority vote per pair
    majority_rows = []
    group_cols = ["pair_id", "dataset", "prompt_id", "prompt", "seed"]

    for keys, g in votes.groupby(group_cols, dropna=False):
        counts = g["preference_method"].value_counts()
        count_dict = {c: int(counts.get(c, 0)) for c in valid_prefs}
        max_count = max(count_dict.values())
        winners = [k for k, v in count_dict.items() if v == max_count]

        if max_count >= 2 and len(winners) == 1:
            majority = winners[0]
        else:
            majority = "no_majority"

        row = dict(zip(group_cols, keys))
        row.update({
            "n_sa_ebama": count_dict["sa_ebama"],
            "n_ebama": count_dict["ebama"],
            "n_tie": count_dict["tie"],
            "n_unclear": count_dict["unclear"],
            "majority_preference": majority,
        })
        majority_rows.append(row)

    majority = pd.DataFrame(majority_rows)
    majority.to_csv(out_dir / "human_preference_majority_vote.csv", index=False, encoding="utf-8-sig")

    # Majority summary
    majority_order = ["sa_ebama", "ebama", "tie", "unclear", "no_majority"]
    majority_counts = majority["majority_preference"].value_counts().reindex(majority_order, fill_value=0)
    total_pairs = int(majority_counts.sum())

    majority_summary_rows = []
    for pref in majority_order:
        k = int(majority_counts[pref])
        ci_low, ci_high = wilson_ci(k, total_pairs)
        majority_summary_rows.append({
            "majority_preference": pref,
            "count": k,
            "rate": k / total_pairs if total_pairs else float("nan"),
            "rate_percent": 100 * k / total_pairs if total_pairs else float("nan"),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })

    decisive_pair_n = int(majority_counts["sa_ebama"] + majority_counts["ebama"])
    decisive_pair_sa = int(majority_counts["sa_ebama"])
    decisive_pair_ebama = int(majority_counts["ebama"])
    p_value_pair = binom_two_sided_p(decisive_pair_sa, decisive_pair_n)

    majority_summary = pd.DataFrame(majority_summary_rows)
    majority_summary.to_csv(out_dir / "human_preference_majority_summary.csv", index=False, encoding="utf-8-sig")

    # By-dataset summary
    by_dataset = (
        majority
        .pivot_table(index="dataset", columns="majority_preference", values="pair_id",
                     aggfunc="count", fill_value=0)
        .reset_index()
    )
    for col in majority_order:
        if col not in by_dataset.columns:
            by_dataset[col] = 0
    by_dataset["n"] = by_dataset[majority_order].sum(axis=1)
    by_dataset["sa_ebama_rate"] = by_dataset["sa_ebama"] / by_dataset["n"]
    by_dataset["ebama_rate"] = by_dataset["ebama"] / by_dataset["n"]
    by_dataset["tie_rate"] = by_dataset["tie"] / by_dataset["n"]
    by_dataset["no_majority_rate"] = by_dataset["no_majority"] / by_dataset["n"]
    by_dataset.to_csv(out_dir / "human_preference_by_dataset.csv", index=False, encoding="utf-8-sig")

    # Agreement
    count_cols = ["n_sa_ebama", "n_ebama", "n_tie", "n_unclear"]
    majority["all_three_same"] = majority[count_cols].max(axis=1) == 3
    majority["has_majority"] = majority["majority_preference"] != "no_majority"

    # Pairwise agreement
    pairwise_same = 0
    pairwise_total = 0
    for _, g in votes.groupby("pair_id"):
        prefs = list(g["preference_method"])
        for i in range(len(prefs)):
            for j in range(i + 1, len(prefs)):
                pairwise_total += 1
                if prefs[i] == prefs[j]:
                    pairwise_same += 1

    kappa_df = majority.rename(columns={
        "n_sa_ebama": "sa_ebama",
        "n_ebama": "ebama",
        "n_tie": "tie",
        "n_unclear": "unclear"
    })
    kappa = fleiss_kappa(kappa_df, valid_prefs)

    agreement = pd.DataFrame([{
        "n_pairs": total_pairs,
        "all_three_same_count": int(majority["all_three_same"].sum()),
        "all_three_same_rate": float(majority["all_three_same"].mean()),
        "has_majority_count": int(majority["has_majority"].sum()),
        "has_majority_rate": float(majority["has_majority"].mean()),
        "pairwise_agreement_count": pairwise_same,
        "pairwise_total": pairwise_total,
        "pairwise_agreement_rate": pairwise_same / pairwise_total if pairwise_total else float("nan"),
        "fleiss_kappa": kappa,
        "decisive_vote_n": decisive_vote_n,
        "decisive_vote_sa_ebama": decisive_vote_sa,
        "decisive_vote_ebama": decisive_vote_ebama,
        "decisive_vote_sa_ebama_rate": decisive_vote_sa / decisive_vote_n if decisive_vote_n else float("nan"),
        "decisive_vote_binom_p": p_value_vote,
        "decisive_pair_n": decisive_pair_n,
        "decisive_pair_sa_ebama": decisive_pair_sa,
        "decisive_pair_ebama": decisive_pair_ebama,
        "decisive_pair_sa_ebama_rate": decisive_pair_sa / decisive_pair_n if decisive_pair_n else float("nan"),
        "decisive_pair_binom_p": p_value_pair,
    }])
    agreement.to_csv(out_dir / "human_preference_agreement.csv", index=False, encoding="utf-8-sig")

    # Paper-ready table
    paper_rows = [
        {
            "Preference outcome": "SA-EBAMA preferred",
            "Majority vote n": int(majority_counts["sa_ebama"]),
            "Rate (%)": 100 * majority_counts["sa_ebama"] / total_pairs,
        },
        {
            "Preference outcome": "EBAMA preferred",
            "Majority vote n": int(majority_counts["ebama"]),
            "Rate (%)": 100 * majority_counts["ebama"] / total_pairs,
        },
        {
            "Preference outcome": "Tie",
            "Majority vote n": int(majority_counts["tie"]),
            "Rate (%)": 100 * majority_counts["tie"] / total_pairs,
        },
        {
            "Preference outcome": "No majority / unresolved",
            "Majority vote n": int(majority_counts["no_majority"] + majority_counts["unclear"]),
            "Rate (%)": 100 * (majority_counts["no_majority"] + majority_counts["unclear"]) / total_pairs,
        },
        {
            "Preference outcome": "SA-EBAMA among decisive pairs",
            "Majority vote n": f"{decisive_pair_sa} / {decisive_pair_n}",
            "Rate (%)": 100 * decisive_pair_sa / decisive_pair_n if decisive_pair_n else float("nan"),
        },
        {
            "Preference outcome": "EBAMA among decisive pairs",
            "Majority vote n": f"{decisive_pair_ebama} / {decisive_pair_n}",
            "Rate (%)": 100 * decisive_pair_ebama / decisive_pair_n if decisive_pair_n else float("nan"),
        },
    ]
    paper_table = pd.DataFrame(paper_rows)
    paper_table.to_csv(out_dir / "paper_human_preference_table.csv", index=False, encoding="utf-8-sig")

    if args.make_figures:
        # Majority vote figure
        fig_data = majority_summary[majority_summary["majority_preference"].isin(
            ["sa_ebama", "ebama", "tie", "no_majority"]
        )].copy()
        labels = ["SA-EBAMA", "EBAMA", "Tie", "No majority"]
        vals = [
            majority_counts["sa_ebama"],
            majority_counts["ebama"],
            majority_counts["tie"],
            majority_counts["no_majority"],
        ]

        plt.figure(figsize=(5.2, 3.4))
        bars = plt.bar(labels, vals)
        plt.ylabel("Number of prompt-seed pairs")
        plt.title("Paired Human Preference: Majority Vote")
        for bar, v in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(v)),
                     ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "human_preference_majority_bar.pdf", bbox_inches="tight")
        plt.savefig(out_dir / "human_preference_majority_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Vote-level figure
        vals2 = [
            vote_counts["sa_ebama"],
            vote_counts["ebama"],
            vote_counts["tie"],
            vote_counts["unclear"],
        ]
        labels2 = ["SA-EBAMA", "EBAMA", "Tie", "Unclear"]

        plt.figure(figsize=(5.2, 3.4))
        bars = plt.bar(labels2, vals2)
        plt.ylabel("Number of annotator votes")
        plt.title("Paired Human Preference: Annotator Votes")
        for bar, v in zip(bars, vals2):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(v)),
                     ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "human_preference_vote_bar.pdf", bbox_inches="tight")
        plt.savefig(out_dir / "human_preference_vote_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nDone. Outputs saved to:", out_dir)
    print("\nVote-level preference counts:")
    print(vote_counts)
    print("\nMajority-vote preference counts:")
    print(majority_counts)
    print("\nDecisive majority pairs:")
    print(f"SA-EBAMA: {decisive_pair_sa}/{decisive_pair_n} = {decisive_pair_sa/decisive_pair_n:.4f}" if decisive_pair_n else "No decisive pairs")
    print(f"Exact binomial p-value among decisive pairs: {p_value_pair:.6g}" if decisive_pair_n else "")


if __name__ == "__main__":
    main()
