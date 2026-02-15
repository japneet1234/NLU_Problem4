# importing necessary libraries
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


# Function to build dataset from CSV file
def build_dataset(data_path: Path) -> Dict[str, List[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    
    # for AG News format (class, title, description)
    if list(df.columns[:3]) == [0, 1, 2] or 'Class Index' in str(df.columns):
        df.columns = ['class', 'title', 'description']
        # Map: Class 1=World->politics, Class 2=Sports->sports
        df = df[df['class'].isin([1, 2])].copy()
        df['class'] = df['class'].map({1: "politics", 2: "sports"})
        df['text'] = df['title'] + " " + df['description']
        return {
            "texts": df["text"].tolist(),
            "labels": df["class"].tolist(),
            "target_names": ["politics", "sports"],
        }
    
    # for handling BBC format (text, category)
    elif "text" in df.columns and "category" in df.columns:
        df = df[df["category"].isin(["sport", "politics"])].copy()
        df["category"] = df["category"].map({"sport": "sports", "politics": "politics"})
        return {
            "texts": df["text"].tolist(),
            "labels": df["category"].tolist(),
            "target_names": ["sports", "politics"],
        }
    
    else:

        # if format is not recognized, raise an error
        raise ValueError("Dataset format not recognized. Expected either:\n"
                        "AG News: columns [0, 1, 2] or 'Class Index'\n"
                        "BBC: columns 'text' and 'category'")


# Function to get vectorizers (BoW and TF-IDF with unigrams and bigrams)
def get_vectorizers():
    return [
        ("bow_unigram", CountVectorizer(ngram_range=(1, 1), min_df=2)),
        ("bow_bigram", CountVectorizer(ngram_range=(1, 2), min_df=2)),
        ("tfidf_unigram", TfidfVectorizer(ngram_range=(1, 1), min_df=2)),
        ("tfidf_bigram", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ]

# Function to get models (Logistic Regression, KNN, SVM, Naive Bayes, Random Forest)
def get_models():
    return [
        ("log_reg", LogisticRegression(max_iter=2000, n_jobs=None)),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ("svm", LinearSVC(max_iter=5000)),
        ("naive_bayes", MultinomialNB()),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]

# Function to evaluate all vectorizer + model combinations and return results
def evaluate(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []

    for vec_name, vectorizer in get_vectorizers():
        for model_name, model in get_models():
            pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("model", model),
            ])
            
            # Train the model and make predictions
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics and store results
            results.append({
                "vectorizer": vec_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            })

    return pd.DataFrame(results)

# Function to display outputs and print results
def save_outputs(df: pd.DataFrame, dataset_stats: Dict[str, int]) -> None:
    
    # Dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    for key, value in dataset_stats.items():
        print(f"{key}: {value}")
    
    # Configuration rankings
    print("\n" + "="*80)
    print("DETAILED RESULTS - ALL CONFIGURATIONS")
    print("="*80)
    df_sorted = df.sort_values(by=["f1_macro", "accuracy"], ascending=False)
    print(df_sorted.to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS")
    print("="*80)
    top_5 = df_sorted.head(5)
    print(top_5.to_string(index=False))


# Main function to run the training and evaluation pipeline
def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate sports vs politics classifiers.")
    
    # Added argument for dataset path 
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("bbc-text") / "bbc-text.csv",
        help="Path to the dataset CSV (expects columns: text, category).",
    )
    # Added stratify=y to ensure class distribution is maintained in train/test split
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    # Build dataset and split into train/test
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
    
    # Dataset statistics for analysis
    dataset_stats = {
        "total_documents": int(len(y)),
        "sports": int(np.sum(y == "sports")),
        "politics": int(np.sum(y == "politics")),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }
    
    # Evaluate models and displaying outputs
    print("Training models\n")
    df = evaluate(X_train, X_test, y_train, y_test)
    save_outputs(df, dataset_stats)
    
    # Print best configuration among all combinations
    best_row = df.sort_values(by="f1_macro", ascending=False).iloc[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(best_row.to_string())

# Run the main function
if __name__ == "__main__":
    main()