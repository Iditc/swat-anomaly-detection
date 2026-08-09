"""LSTM Autoencoder — learns normal temporal patterns, flags deviations."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
)
from tensorflow import keras
from tensorflow.keras import layers

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

SEQ_LEN = 60
RANDOM_STATE = 42


def load_data():
    """Load featured train/test data."""
    train = pd.read_parquet(PROCESSED_DIR / "train_featured.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_featured.parquet")
    return train, test


def get_feature_columns(df):
    """Return numeric feature columns."""
    exclude = {"Timestamp", "label"}
    return [c for c in df.columns if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])]


def make_sequences(X, seq_len, stride=1):
    """Convert array to sequences with stride for memory efficiency."""
    n_seqs = (len(X) - seq_len) // stride + 1
    seqs = np.empty((n_seqs, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n_seqs):
        start = i * stride
        seqs[i] = X[start:start + seq_len]
    return seqs


def build_model(seq_len, n_features):
    """Build LSTM Autoencoder."""
    encoder_input = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(encoder_input)
    x = layers.LSTM(32, return_sequences=False)(x)
    bottleneck = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(32, return_sequences=True)(bottleneck)
    x = layers.LSTM(64, return_sequences=True)(x)
    output = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = keras.Model(encoder_input, output)
    model.compile(optimizer="adam", loss="mse")
    return model


def score_batched(model, X, batch_size=512):
    """Score sequences in batches to save memory."""
    scores = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        recon = model.predict(batch, verbose=0)
        scores[i:i + batch_size] = np.mean((batch - recon) ** 2, axis=(1, 2))
    return scores


def best_threshold_predictions(y_true, scores):
    """Find threshold maximizing F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall), 0,
    )
    best_idx = np.argmax(f1)
    threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    return (scores >= threshold).astype(int), threshold


def main():
    train_df, test_df = load_data()
    feature_cols = get_feature_columns(train_df)
    n_features = len(feature_cols)

    X_train_normal = train_df[train_df["label"] == 0][feature_cols].values
    X_test_raw = test_df[feature_cols].values
    y_test_full = test_df["label"].values

    print(f"Train normal: {len(X_train_normal):,}, Test: {len(X_test_raw):,}")
    print(f"Features: {n_features}, Sequence length: {SEQ_LEN}")

    X_train_normal = X_train_normal.astype(np.float32)
    X_test_raw = X_test_raw.astype(np.float32)

    train_stride = 10
    print(f"\nBuilding train sequences (stride={train_stride})...")
    train_seqs = make_sequences(X_train_normal, SEQ_LEN, stride=train_stride)
    print(f"  Train sequences: {train_seqs.shape[0]:,} ({train_seqs.nbytes/1e9:.1f}GB)")

    print("\nBuilding LSTM Autoencoder...")
    model = build_model(SEQ_LEN, n_features)
    model.summary()

    print("\nTraining...")
    model.fit(
        train_seqs, train_seqs,
        epochs=30, batch_size=256,
        validation_split=0.1, verbose=1,
    )
    del train_seqs

    print("\nScoring test (batch-by-batch to save memory)...")
    test_stride = 5
    test_seqs = make_sequences(X_test_raw, SEQ_LEN, stride=test_stride)
    y_test_strided = y_test_full[SEQ_LEN - 1::test_stride][:len(test_seqs)]
    print(f"  Test sequences: {test_seqs.shape[0]:,} ({test_seqs.nbytes/1e9:.1f}GB)")

    scores = score_batched(model, test_seqs)
    del test_seqs
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")

    y_pred, threshold = best_threshold_predictions(y_test_strided, scores)
    f1_macro = f1_score(y_test_strided, y_pred, average="macro")
    report = classification_report(
        y_test_strided, y_pred, target_names=["Normal", "Attack"],
        labels=[0, 1], output_dict=True,
    )

    print(f"\n{'='*55}")
    print("  LSTM AUTOENCODER — Results")
    print(f"{'='*55}")
    print(f"  F1 Macro:    {f1_macro:.4f}")
    print(f"  Precision:   {report['Attack']['precision']:.4f}")
    print(f"  Recall:      {report['Attack']['recall']:.4f}")
    print(f"  Threshold:   {threshold:.6f}")

    attack_indices = np.where(y_test_strided == 1)[0]
    if len(attack_indices) > 0:
        breaks = np.diff(attack_indices)
        period_ids = np.zeros(len(attack_indices), dtype=int)
        pid = 0
        for i in range(len(attack_indices)):
            if i > 0 and breaks[i - 1] > 1:
                pid += 1
            period_ids[i] = pid

        print("\n  Per attack period:")
        for p in range(period_ids.max() + 1):
            mask = period_ids == p
            period_pred = y_pred[attack_indices[mask]]
            duration = mask.sum()
            caught = period_pred.sum()
            rate = caught / duration if duration > 0 else 0
            was_missed = "  (was MISSED)" if p < 3 else ""
            print(f"    P{p+1}: {int(caught):>4}/{duration} "
                  f"({rate:>5.1%}){was_missed}")

    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    metrics = {
        "f1_macro": round(f1_macro, 4),
        "precision_attack": round(report["Attack"]["precision"], 4),
        "recall_attack": round(report["Attack"]["recall"], 4),
        "threshold": round(threshold, 6),
    }
    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "metrics" / "lstm_autoencoder.csv", index=False,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_autoencoder.keras")
    print(f"\nModel saved: {MODELS_DIR / 'lstm_autoencoder.keras'}")


if __name__ == "__main__":
    main()
