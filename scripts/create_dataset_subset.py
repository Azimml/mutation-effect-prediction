from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified dataset subset from a processed CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input_csv, dtype={"chrom": "string"}, low_memory=False)
    if args.target_rows <= 0 or args.target_rows > len(dataframe):
        raise ValueError(f"--target-rows must be between 1 and {len(dataframe)}.")

    subset = stratified_subset(
        dataframe=dataframe,
        target_rows=args.target_rows,
        seed=args.seed,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False, compression="gzip")
    print(f"Wrote {len(subset)} rows to {output_path}")


def stratified_subset(dataframe: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    group_keys = ["split", "label_name"]
    group_sizes = dataframe.groupby(group_keys, observed=True).size()
    fractions = group_sizes / len(dataframe)
    requested = (fractions * target_rows).round().astype(int)

    difference = target_rows - int(requested.sum())
    if difference != 0:
        ordering = (fractions * target_rows - requested).sort_values(ascending=(difference < 0))
        for group_key in ordering.index:
            if difference == 0:
                break
            step = 1 if difference > 0 else -1
            if requested[group_key] + step < 1:
                continue
            requested[group_key] += step
            difference -= step

    subset_frames: list[pd.DataFrame] = []
    for group_key, frame in dataframe.groupby(group_keys, observed=True):
        n_rows = min(int(requested[group_key]), len(frame))
        subset_frames.append(frame.sample(n=n_rows, random_state=seed))

    subset = pd.concat(subset_frames, ignore_index=True)
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    main()
