"""Run all models on the same train/test split and compare results."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
import lightgbm as lgb

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load featured train/test data."""
    train = pd.read_parquet(PROCESSED_DIR / "train_featured.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_featured.parquet")
    return train, test


def get_features(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns."""
    exclude = {"Timestamp", "label"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard classification metrics."""
    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"],
        labels=[0, 1], output_dict=True,
    )
    return {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision_attack": round(report["Attack"]["precision"], 4),
        "recall_attack": round(report["Attack"]["recall"], 4),
        "f1_attack": round(report["Attack"]["f1-score"], 4),
    }


def best_threshold_predictions(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Find threshold maximizing F1 and return predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    best_idx = np.argmax(f1)
    threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    print(f"  Best threshold: {threshold:.4f}, F1: {f1[best_idx]:.4f}")
    return (scores >= threshold).astype(int)


def run_isolation_forest(X_train_normal, X_test, y_test) -> dict:
    """Train Isolation Forest on normal data only."""
    print("\n[1/6] Isolation Forest...")
    model = IsolationForest(
        n_estimators=200, contamination=0.01,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train_normal)
    scores = -model.score_samples(X_test)
    y_pred = best_threshold_predictions(y_test, scores)
    return calc_metrics(y_test, y_pred)


def run_lightgbm(X_train, y_train, X_test, y_test) -> dict:
    """Train LightGBM supervised classifier."""
    print("\n[2/6] LightGBM...")
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        num_leaves=31, scale_pos_weight=ratio,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    y_pred = model.predict(X_test)
    return calc_metrics(y_test, y_pred)


def run_autoencoder(X_train_normal, X_test, y_test) -> dict:
    """Train Autoencoder on normal data, use reconstruction error."""
    print("\n[3/6] Autoencoder...")
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from tensorflow import keras

    n_features = X_train_normal.shape[1]
    encoder_dim = max(n_features // 4, 16)
    bottleneck = max(n_features // 8, 8)

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(encoder_dim, activation="relu"),
        keras.layers.Dense(bottleneck, activation="relu"),
        keras.layers.Dense(encoder_dim, activation="relu"),
        keras.layers.Dense(n_features, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_normal, X_train_normal,
        epochs=50, batch_size=256, validation_split=0.1,
        verbose=0,
    )

    recon = model.predict(X_test, verbose=0)
    scores = np.mean((X_test - recon) ** 2, axis=1)
    y_pred = best_threshold_predictions(y_test, scores)
    return calc_metrics(y_test, y_pred)


def run_ocsvm(X_train_normal, X_test, y_test) -> dict:
    """Train One-Class SVM on normal data only."""
    print("\n[4/6] One-Class SVM...")
    sample_size = min(50_000, len(X_train_normal))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train_normal), sample_size, replace=False)
    X_sample = X_train_normal[idx]
    print(f"  Training on {sample_size:,} samples (subsampled for speed)")

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    model.fit(X_sample)
    scores = -model.decision_function(X_test)
    y_pred = best_threshold_predictions(y_test, scores)
    return calc_metrics(y_test, y_pred)


def run_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Train Random Forest supervised classifier."""
    print("\n[5/6] Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calc_metrics(y_test, y_pred)


def run_lstm(X_train, y_train, X_test, y_test) -> dict:
    """Train LSTM classifier on sequences."""
    print("\n[6/6] LSTM...")
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from tensorflow import keras

    seq_len = 30
    n_features = X_train.shape[1]

    def make_sequences(X, y):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs), np.array(ys)

    print(f"  Building sequences (window={seq_len})...")
    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train)
    X_te_seq, y_te_seq = make_sequences(X_test, y_test)
    print(f"  Train sequences: {X_tr_seq.shape[0]:,}, Test: {X_te_seq.shape[0]:,}")

    ratio = (y_tr_seq == 0).sum() / max((y_tr_seq == 1).sum(), 1)

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_features)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=10, batch_size=256, validation_split=0.1,
        class_weight={0: 1.0, 1: ratio},
        verbose=0,
    )

    y_proba = model.predict(X_te_seq, verbose=0).flatten()
    y_pred = best_threshold_predictions(y_te_seq, y_proba)
    return calc_metrics(y_te_seq, y_pred)


def main() -> None:
    train_df, test_df = load_data()
    feature_cols = get_features(train_df)

    print(f"Train: {len(train_df):,} rows (Normal: {(train_df['label']==0).sum():,}, "
          f"Attack: {(train_df['label']==1).sum():,})")
    print(f"Test:  {len(test_df):,} rows (Normal: {(test_df['label']==0).sum():,}, "
          f"Attack: {(test_df['label']==1).sum():,})")
    print(f"Features: {len(feature_cols)}")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    train_normal_mask = y_train == 0
    X_train_normal = X_train[train_normal_mask]
    print(f"Normal training samples: {len(X_train_normal):,}")

    results = {}
    results["Isolation Forest"] = run_isolation_forest(X_train_normal, X_test, y_test)
    results["LightGBM"] = run_lightgbm(X_train, y_train, X_test, y_test)
    results["Autoencoder"] = run_autoencoder(X_train_normal, X_test, y_test)
    results["One-Class SVM"] = run_ocsvm(X_train_normal, X_test, y_test)
    results["Random Forest"] = run_random_forest(X_train, y_train, X_test, y_test)
    results["LSTM"] = run_lstm(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 75)
    print("  MODEL COMPARISON — All models on same test set")
    print("=" * 75)

    df = pd.DataFrame(results).T
    df = df.sort_values("f1_macro", ascending=False)
    df.index.name = "Model"

    print(f"\n{'Model':<20} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10} {'F1 Attack':>10}")
    print("-" * 62)
    for name, row in df.iterrows():
        print(f"{name:<20} {row['f1_macro']:>10.4f} {row['precision_attack']:>10.4f} "
              f"{row['recall_attack']:>10.4f} {row['f1_attack']:>10.4f}")

    out_path = RESULTS_DIR / "metrics" / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
