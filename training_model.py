from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_dataset(data_path: Path) -> Dict[str, List[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("Expected columns 'text' and 'category' in the BBC dataset CSV.")

    df = df[df["category"].isin(["sport", "politics"])].copy()
    df["category"] = df["category"].map({"sport": "sports", "politics": "politics"})

    return {
        "texts": df["text"].tolist(),
        "labels": df["category"].tolist(),
        "target_names": ["sports", "politics"],
    }


def get_vectorizers():
    return [
        ("bow_unigram", CountVectorizer(ngram_range=(1, 1), min_df=2)),
        ("bow_bigram", CountVectorizer(ngram_range=(1, 2), min_df=2)),
        ("tfidf_unigram", TfidfVectorizer(ngram_range=(1, 1), min_df=2)),
        ("tfidf_bigram", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ]


def get_models():
    return [
        ("log_reg", LogisticRegression(max_iter=2000, n_jobs=None)),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ("svm", LinearSVC(max_iter=5000)),
        ("naive_bayes", MultinomialNB()),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]


def evaluate(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []

    for vec_name, vectorizer in get_vectorizers():
        for model_name, model in get_models():
            pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("model", model),
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            results.append({
                "vectorizer": vec_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            })

    return pd.DataFrame(results)


def save_outputs(df: pd.DataFrame, dataset_stats: Dict[str, int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values(by=["f1_macro", "accuracy"], ascending=False)
    df_sorted.to_csv(output_dir / "metrics.csv", index=False)

    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate sports vs politics classifiers.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("bbc-text") / "bbc-text.csv",
        help="Path to the BBC dataset CSV (expects columns: text, category).",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    data = build_dataset(args.data_path)
    X = np.array(data["texts"], dtype=object)
    y = np.array(data["labels"], dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    dataset_stats = {
        "total_documents": int(len(y)),
        "sports": int(np.sum(y == "sports")),
        "politics": int(np.sum(y == "politics")),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }

    df = evaluate(X_train, X_test, y_train, y_test)
    save_outputs(df, dataset_stats, args.output_dir)

    best_row = df.sort_values(by="f1_macro", ascending=False).iloc[0]
    print("Best configuration:")
    print(best_row.to_string())


if __name__ == "__main__":
    main()
