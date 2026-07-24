from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .dataframe_io import read_dataset_csv, require_columns
from .evaluation import compute_binary_metrics, save_classification_plots
from .utils import ensure_directory, write_json


def train_baseline_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    kmer_size: int,
    max_features: int,
    max_iter: int,
    class_weight: str | None,
) -> dict[str, dict[str, float]]:
    dataframe = read_dataset_csv(dataset_path)
    train_frame = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
    val_frame = dataframe[dataframe["split"] == "val"].reset_index(drop=True)
    test_frame = dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    validate_split_frame(train_frame, "train")
    validate_split_frame(val_frame, "val")
    validate_split_frame(test_frame, "test")

    vectorizer = TfidfVectorizer(analyzer="word", max_features=max_features)
    classifier = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        solver="saga",
        random_state=17,
    )

    train_texts = build_text_features(train_frame, kmer_size)
    vectorizer.fit(train_texts)
    train_matrix = vectorizer.transform(train_texts)
    classifier.fit(train_matrix, train_frame["label"].to_numpy())

    metrics_by_split: dict[str, dict[str, float]] = {}
    output_directory = ensure_directory(output_dir)

    for split_name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        texts = build_text_features(frame, kmer_size)
        matrix = vectorizer.transform(texts)
        scores = classifier.predict_proba(matrix)[:, 1]
        labels = frame["label"].to_numpy()
        metrics_by_split[split_name] = compute_binary_metrics(labels, scores)
        save_classification_plots(labels, scores, output_directory / "plots", prefix=f"baseline_{split_name}")

    with (output_directory / "baseline_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "kmer_size": kmer_size,
            },
            handle,
        )

    write_json(metrics_by_split, output_directory / "metrics.json")
    return metrics_by_split


def predict_with_baseline(
    model_path: str | Path,
    input_csv_path: str | Path,
    output_csv_path: str | Path,
) -> Path:
    with Path(model_path).open("rb") as handle:
        payload = pickle.load(handle)

    dataframe = read_dataset_csv(input_csv_path)
    require_columns(dataframe, ("ref_seq", "alt_seq"), input_csv_path)
    texts = build_text_features(dataframe, payload["kmer_size"])
    matrix = payload["vectorizer"].transform(texts)
    scores = payload["classifier"].predict_proba(matrix)[:, 1]

    result = dataframe.copy()
    result["predicted_probability"] = scores
    result["predicted_label"] = (scores >= 0.5).astype(int)

    output_path = Path(output_csv_path)
    ensure_directory(output_path.parent)
    result.to_csv(output_path, index=False)
    return output_path


def build_text_features(dataframe: pd.DataFrame, kmer_size: int) -> list[str]:
    return [
        " ".join(sequence_to_kmers(ref_seq, kmer_size))
        + " [ALT] "
        + " ".join(sequence_to_kmers(alt_seq, kmer_size))
        for ref_seq, alt_seq in zip(dataframe["ref_seq"], dataframe["alt_seq"], strict=True)
    ]


def sequence_to_kmers(sequence: str, kmer_size: int) -> list[str]:
    if len(sequence) < kmer_size:
        return [sequence]

    return [sequence[index : index + kmer_size] for index in range(len(sequence) - kmer_size + 1)]


def validate_split_frame(dataframe: pd.DataFrame, split_name: str) -> None:
    if dataframe.empty:
        raise ValueError(f"Split '{split_name}' is empty. Increase dataset size or adjust split settings.")

    if dataframe["label"].nunique() < 2:
        raise ValueError(
            f"Split '{split_name}' does not contain both classes. Increase dataset size or adjust split settings."
        )
