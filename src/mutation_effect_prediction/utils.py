"""Small filesystem, download, hashing, and seeding helpers.

Deliberately dependency-light: numpy and torch are imported lazily inside
``set_seed`` so the rest of the module works without them installed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write ``data`` as pretty, key-sorted JSON, creating parent dirs."""
    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_file(url: str, destination: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    """Stream ``url`` to ``destination``, downloading via a ``.part`` file.

    Writing to a temporary ``.part`` path and renaming on completion makes the
    download atomic: an interrupted transfer never leaves a truncated file at
    the final path.
    """
    output_path = Path(destination)
    ensure_directory(output_path.parent)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "mutation-effect-prediction/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)

    temp_path.replace(output_path)
    return output_path


def decompress_gzip(source: str | Path, destination: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    """Stream-decompress a gzip file to ``destination`` and return its path."""
    source_path = Path(source)
    destination_path = Path(destination)
    ensure_directory(destination_path.parent)

    with gzip.open(source_path, "rb") as input_handle, destination_path.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=chunk_size)

    return destination_path


def stable_fraction(identifier: str) -> float:
    """Map a string to a deterministic float in ``[0, 1)`` via SHA-1.

    Unlike ``hash()``, this is stable across processes and Python versions,
    which is what makes the region-based train/val/test split reproducible.
    """
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def set_seed(seed: int) -> None:
    """Seed ``random``, and ``numpy``/``torch`` if they are installed."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
