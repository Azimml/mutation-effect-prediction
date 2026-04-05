from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_dataset_csv(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"chrom": "string"},
        low_memory=False,
    )
