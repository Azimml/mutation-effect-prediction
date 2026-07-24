from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def read_dataset_csv(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"chrom": "string"},
        low_memory=False,
    )


def require_columns(dataframe: pd.DataFrame, columns: Iterable[str], source: str | Path) -> None:
    """Raise a clear error if any required column is missing from ``dataframe``.

    This turns an otherwise cryptic downstream ``KeyError`` into an actionable
    message that names the file and the missing columns.
    """
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {', '.join(missing)}. "
            f"Found columns: {', '.join(map(str, dataframe.columns))}."
        )
