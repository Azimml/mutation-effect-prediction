from __future__ import annotations

import subprocess

import torch


def resolve_device(device_preference: str) -> torch.device:
    if device_preference == "cpu":
        return torch.device("cpu")

    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def query_gpu_temperature_c(device_index: int = 0) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    output = result.stdout.strip().splitlines()
    if not output:
        return None

    try:
        return int(output[0].strip())
    except ValueError:
        return None
