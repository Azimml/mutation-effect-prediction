from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataframe_io import read_dataset_csv, require_columns
from .evaluation import (
    compute_binary_metrics,
    save_classification_plots,
    save_training_history,
    select_best_f1_threshold,
)
from .runtime import query_gpu_temperature_c, resolve_device
from .utils import ensure_directory, set_seed, write_json

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


class SequencePairDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        features = encode_sequence_pair(row["ref_seq"], row["alt_seq"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return features, label


class SequencePairCNN(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=9, padding=4),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        center_index = features.shape[-1] // 2
        center_features = features[:, :, center_index]
        max_features = torch.amax(features, dim=-1)
        mean_features = torch.mean(features, dim=-1)
        combined = torch.cat([center_features, max_features, mean_features], dim=1)
        return self.classifier(combined).squeeze(-1)


def train_cnn_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_dim: int,
    early_stopping_patience: int,
    seed: int,
    device_preference: str = "auto",
    resume_from_checkpoint: str | Path | None = None,
    max_temperature_c: int | None = None,
    cooldown_temperature_c: int | None = None,
    temperature_check_interval_batches: int = 25,
    batch_sleep_seconds: float = 0.0,
) -> dict[str, dict[str, float]]:
    set_seed(seed)

    dataframe = read_dataset_csv(dataset_path)
    train_frame = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
    val_frame = dataframe[dataframe["split"] == "val"].reset_index(drop=True)
    test_frame = dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    validate_split_frame(train_frame, "train")
    validate_split_frame(val_frame, "val")
    validate_split_frame(test_frame, "test")

    train_dataset = SequencePairDataset(train_frame)
    val_dataset = SequencePairDataset(val_frame)
    test_dataset = SequencePairDataset(test_frame)

    device = resolve_device(device_preference)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = SequencePairCNN(hidden_dim=hidden_dim, dropout=dropout).to(device)

    positive_count = max(1, int(train_frame["label"].sum()))
    negative_count = max(1, int(len(train_frame) - positive_count))
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, device=device)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, device=device)

    output_directory = ensure_directory(output_dir)
    history: list[dict[str, float]] = []
    best_val_auroc = -math.inf
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1

    latest_checkpoint_path = output_directory / "latest_checkpoint.pt"
    if resume_from_checkpoint is not None:
        checkpoint = load_cnn_checkpoint(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        history = list(checkpoint.get("history", []))
        best_val_auroc = float(checkpoint.get("best_val_auroc", best_val_auroc))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        patience_counter = int(checkpoint.get("patience_counter", patience_counter))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming CNN training from epoch {start_epoch}.", flush=True)

    cooldown_temperature_c = (
        max_temperature_c - 4 if max_temperature_c is not None and cooldown_temperature_c is None else cooldown_temperature_c
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0

        for batch_index, (features, labels) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.shape[0]
            running_loss += loss.item() * batch_size_actual
            running_examples += batch_size_actual

            if batch_sleep_seconds > 0:
                time.sleep(batch_sleep_seconds)

            maybe_cool_gpu(
                device=device,
                batch_index=batch_index,
                temperature_check_interval_batches=temperature_check_interval_batches,
                max_temperature_c=max_temperature_c,
                cooldown_temperature_c=cooldown_temperature_c,
            )

        train_loss = running_loss / max(1, running_examples)
        val_labels, val_scores, val_loss = evaluate_model(model, val_loader, criterion, device)
        val_metrics = compute_binary_metrics(val_labels, val_scores)
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
        }
        history.append(history_entry)
        print(
            f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auroc={val_metrics['auroc']:.4f}",
            flush=True,
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                path=output_directory / "best_checkpoint.pt",
                model=model,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epoch=epoch,
                optimizer=optimizer,
                history=history,
                best_epoch=best_epoch,
                best_val_auroc=best_val_auroc,
                patience_counter=patience_counter,
            )
        else:
            patience_counter += 1

        save_checkpoint(
            path=latest_checkpoint_path,
            model=model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epoch=epoch,
            optimizer=optimizer,
            history=history,
            best_epoch=best_epoch,
            best_val_auroc=best_val_auroc,
            patience_counter=patience_counter,
        )

        if patience_counter >= early_stopping_patience:
            break

    checkpoint = load_cnn_checkpoint(output_directory / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_labels, val_scores, _ = evaluate_model(model, val_loader, criterion, device)
    selected_threshold = select_best_f1_threshold(val_labels, val_scores)
    save_checkpoint(
        path=output_directory / "best_checkpoint.pt",
        model=model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epoch=best_epoch,
        optimizer=optimizer,
        history=history,
        best_epoch=best_epoch,
        selected_threshold=selected_threshold,
        best_val_auroc=best_val_auroc,
        patience_counter=patience_counter,
    )

    metrics_by_split: dict[str, dict[str, float]] = {}
    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        labels, scores, _ = evaluate_model(model, loader, criterion, device)
        metrics_by_split[split_name] = compute_binary_metrics(labels, scores, threshold=selected_threshold)
        save_classification_plots(labels, scores, output_directory / "plots", prefix=f"cnn_{split_name}")

    metrics_by_split["training"] = {
        "best_epoch": float(best_epoch),
        "best_val_auroc": float(best_val_auroc),
        "selected_threshold": float(selected_threshold),
    }
    write_json(metrics_by_split, output_directory / "metrics.json")
    save_training_history(history, output_directory / "training_history.png")
    return metrics_by_split


def predict_with_cnn(
    checkpoint_path: str | Path,
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    device_preference: str = "auto",
) -> Path:
    dataframe = read_dataset_csv(input_csv_path)
    require_columns(dataframe, ("ref_seq", "alt_seq"), input_csv_path)
    dataset = SequencePairDataset(dataframe.assign(label=0))

    device = resolve_device(device_preference)
    checkpoint = load_cnn_checkpoint(checkpoint_path, map_location=device)
    model = SequencePairCNN(hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = build_dataloader(dataset, batch_size=512, shuffle=False, device=device)

    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device, non_blocking=True))
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)

    output = dataframe.copy()
    probabilities = np.concatenate(all_scores)
    selected_threshold = float(checkpoint.get("selected_threshold", 0.5))
    output["predicted_probability"] = probabilities
    output["predicted_label"] = (probabilities >= selected_threshold).astype(int)

    output_path = Path(output_csv_path)
    ensure_directory(output_path.parent)
    output.to_csv(output_path, index=False)
    return output_path


def save_saliency_plot(
    checkpoint_path: str | Path,
    input_csv_path: str | Path,
    variant_id: str,
    output_path: str | Path,
    device_preference: str = "auto",
) -> Path:
    dataframe = read_dataset_csv(input_csv_path)
    selected = dataframe[dataframe["variant_id"] == variant_id]
    if selected.empty:
        raise ValueError(f"Variant '{variant_id}' was not found in {input_csv_path}.")

    row = selected.iloc[0]
    input_tensor = encode_sequence_pair(row["ref_seq"], row["alt_seq"]).unsqueeze(0)

    device = resolve_device(device_preference)
    checkpoint = load_cnn_checkpoint(checkpoint_path, map_location=device)
    model = SequencePairCNN(hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device_tensor = input_tensor.to(device)
    device_tensor.requires_grad_(True)
    logits = model(device_tensor)
    logits.backward(torch.ones_like(logits))

    gradients = device_tensor.grad.detach().cpu().numpy()[0]
    saliency = np.abs(gradients[:4]).sum(axis=0) + np.abs(gradients[4:]).sum(axis=0)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(saliency)
    axis.axvline(len(row["ref_seq"]) // 2, color="crimson", linestyle="--", label="Mutation")
    axis.set_xlabel("Sequence Position")
    axis.set_ylabel("Absolute Gradient")
    axis.set_title(f"Saliency for {variant_id}")
    axis.legend()
    figure.tight_layout()

    destination = Path(output_path)
    ensure_directory(destination.parent)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def encode_sequence_pair(reference_sequence: str, alternate_sequence: str) -> torch.Tensor:
    ref_encoded = one_hot_encode(reference_sequence)
    alt_encoded = one_hot_encode(alternate_sequence)
    mutation_mask = np.zeros((1, len(reference_sequence)), dtype=np.float32)
    mutation_mask[0, len(reference_sequence) // 2] = 1.0
    stacked = np.concatenate([ref_encoded, alt_encoded, mutation_mask], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def one_hot_encode(sequence: str) -> np.ndarray:
    encoded = np.zeros((4, len(sequence)), dtype=np.float32)
    for index, base in enumerate(sequence):
        base_index = BASE_TO_INDEX.get(base)
        if base_index is not None:
            encoded[base_index, index] = 1.0
    return encoded


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            scores = torch.sigmoid(logits)

            batch_size_actual = labels.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    labels = np.concatenate(all_labels) if all_labels else np.array([])
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    average_loss = total_loss / max(1, total_examples)
    return labels, scores, average_loss


def save_checkpoint(
    path: str | Path,
    model: SequencePairCNN,
    hidden_dim: int,
    dropout: float,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    history: list[dict[str, float]] | None = None,
    best_epoch: int | None = None,
    selected_threshold: float | None = None,
    best_val_auroc: float | None = None,
    patience_counter: int | None = None,
) -> None:
    destination = Path(path)
    ensure_directory(destination.parent)
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if history is not None:
        payload["history"] = history
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    if selected_threshold is not None:
        payload["selected_threshold"] = float(selected_threshold)
    if best_val_auroc is not None:
        payload["best_val_auroc"] = float(best_val_auroc)
    if patience_counter is not None:
        payload["patience_counter"] = int(patience_counter)
    torch.save(payload, destination)


def load_cnn_checkpoint(path: str | Path, map_location: str | torch.device) -> dict[str, object]:
    # weights_only=True refuses to unpickle arbitrary objects (an RCE vector on
    # untrusted checkpoints; the torch>=2.6 default). Our payload is only
    # tensors + plain dict/float/int/list, so this is safe and future-proof.
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def validate_split_frame(dataframe: pd.DataFrame, split_name: str) -> None:
    if dataframe.empty:
        raise ValueError(f"Split '{split_name}' is empty. Increase dataset size or adjust split settings.")

    if dataframe["label"].nunique() < 2:
        raise ValueError(
            f"Split '{split_name}' does not contain both classes. Increase dataset size or adjust split settings."
        )


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    worker_cap = 4
    num_workers = min(worker_cap, max(0, (os.cpu_count() or 1) - 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def maybe_cool_gpu(
    device: torch.device,
    batch_index: int,
    temperature_check_interval_batches: int,
    max_temperature_c: int | None,
    cooldown_temperature_c: int | None,
) -> None:
    if device.type != "cuda" or max_temperature_c is None:
        return

    if temperature_check_interval_batches <= 0 or batch_index % temperature_check_interval_batches != 0:
        return

    current_temperature_c = query_gpu_temperature_c()
    if current_temperature_c is None or current_temperature_c < max_temperature_c:
        return

    cooldown_target = cooldown_temperature_c if cooldown_temperature_c is not None else max_temperature_c - 4
    print(
        f"GPU reached {current_temperature_c}C at batch {batch_index}; pausing until it cools to {cooldown_target}C.",
        flush=True,
    )
    while True:
        time.sleep(10)
        current_temperature_c = query_gpu_temperature_c()
        if current_temperature_c is None:
            return
        if current_temperature_c <= cooldown_target:
            print(f"GPU cooled to {current_temperature_c}C; resuming training.", flush=True)
            return
