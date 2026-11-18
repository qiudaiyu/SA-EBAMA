

import argparse
from pathlib import Path

import pandas as pd


def resolve_path(project_root: Path, p: str) -> str:
    p = Path(str(p).strip())
    if not p.is_absolute():
        p = project_root / p
    return str(p)


def get_image_path(index_df, project_root: Path, prompt_id: str, seed: int, method_name: str) -> str:
    sub = index_df[
        (index_df["prompt_id"].astype(str).str.strip() == str(prompt_id).strip())
        & (index_df["seed"].astype(int) == int(seed))
        & (index_df["method_norm"] == method_name)
    ].copy()

    if sub.empty:
        available = index_df[
            (index_df["prompt_id"].astype(str).str.strip() == str(prompt_id).strip())
            & (index_df["seed"].astype(int) == int(seed))
        ]["method_norm"].drop_duplicates().tolist()

        raise RuntimeError(
            f"Missing image for prompt_id={prompt_id}, seed={seed}, method={method_name}. "
            f"Available methods for this prompt-seed: {available}"
        )

    if "image_path" not in sub.columns:
        raise RuntimeError("Column image_path not found in all_images_index.csv")

    p = resolve_path(project_root, sub.iloc[0]["image_path"])

    if not Path(p).exists():
        raise FileNotFoundError(
            f"Image path exists in CSV but file is missing for "
            f"prompt_id={prompt_id}, seed={seed}, method={method_name}: {p}"
        )

    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    project_root = Path.cwd()

    examples_path = Path(args.examples_csv)
    index_path = Path(args.index_csv)

    if not examples_path.exists():
        raise FileNotFoundError(f"examples_csv not found: {examples_path}")

    if not index_path.exists():
        raise FileNotFoundError(f"index_csv not found: {index_path}")

    examples = pd.read_csv(examples_path)
    index_df = pd.read_csv(index_path)

    required_example_cols = {"prompt_id", "prompt", "seed"}
    missing_example = required_example_cols - set(examples.columns)
    if missing_example:
        raise RuntimeError(f"Missing required columns in examples_csv: {missing_example}")

    required_index_cols = {"method", "prompt_id", "prompt", "seed", "image_path"}
    missing_index = required_index_cols - set(index_df.columns)
    if missing_index:
        raise RuntimeError(f"Missing required columns in all_images_index.csv: {missing_index}")

    index_df["method_norm"] = index_df["method"].astype(str).str.strip().str.lower()

    rows = []
    for _, r in examples.iterrows():
        prompt_id = str(r["prompt_id"]).strip()
        seed = int(r["seed"])

        row = r.to_dict()

        row["ebama_image"] = get_image_path(
            index_df=index_df,
            project_root=project_root,
            prompt_id=prompt_id,
            seed=seed,
            method_name="ebama",
        )

        row["sa_image"] = get_image_path(
            index_df=index_df,
            project_root=project_root,
            prompt_id=prompt_id,
            seed=seed,
            method_name="sa_ebama",
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("\nAttached image paths:")
    show_cols = [
        "dataset", "prompt_id", "prompt", "seed",
        "ebama_oor", "sa_oor", "delta_oor",
        "ebama_image", "sa_image"
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
