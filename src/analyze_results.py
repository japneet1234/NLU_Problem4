from pathlib import Path
import json
import pandas as pd


def analyze_results(metrics_csv: Path, stats_json: Path) -> None:
    """Load and print comprehensive analysis of results."""
    
    df = pd.read_csv(metrics_csv)
    
    with open(stats_json, 'r') as f:
        stats = json.load(f)
    
    print("=" * 80)
    print("SPORTS vs POLITICS CLASSIFICATION RESULTS ANALYSIS (BBC DATASET)")
    print("=" * 80)
    
    # Dataset Overview
    print("\n[DATASET OVERVIEW]")
    print(f"Total Documents: {stats['total_documents']:,}")
    print(f"Sports: {stats['sports']:,} ({100*stats['sports']/stats['total_documents']:.1f}%)")
    print(f"Politics: {stats['politics']:,} ({100*stats['politics']/stats['total_documents']:.1f}%)")
    print(f"Train Set: {stats['train_size']:,} documents")
    print(f"Test Set: {stats['test_size']:,} documents")
    
    # Top performers
    print("\n[TOP 5 BEST CONFIGURATIONS (by F1-macro)]")
    top_5 = df.nlargest(5, 'f1_macro')[['vectorizer', 'model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]
    for idx, row in top_5.iterrows():
        print(f"  {idx+1}. {row['vectorizer']:20s} + {row['model']:15s} → F1={row['f1_macro']:.4f} (Acc={row['accuracy']:.4f})")
    
    # Bottom performers
    print("\n[BOTTOM 5 CONFIGURATIONS (by F1-macro)]")
    bottom_5 = df.nsmallest(5, 'f1_macro')[['vectorizer', 'model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]
    for idx, row in bottom_5.iterrows():
        print(f"  {idx+1}. {row['vectorizer']:20s} + {row['model']:15s} → F1={row['f1_macro']:.4f} (Acc={row['accuracy']:.4f})")
    
    # Model rankings
    print("\n[MODEL PERFORMANCE (average across vectorizers)]")
    model_avg = df.groupby('model')[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']].mean()
    model_avg = model_avg.sort_values('f1_macro', ascending=False)
    for model, row in model_avg.iterrows():
        print(f"  {model:20s}: F1={row['f1_macro']:.4f} | Acc={row['accuracy']:.4f} | Prec={row['precision_macro']:.4f} | Rec={row['recall_macro']:.4f}")
    
    # Vectorizer rankings
    print("\n[VECTORIZER PERFORMANCE (average across models)]")
    vec_avg = df.groupby('vectorizer')[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']].mean()
    vec_avg = vec_avg.sort_values('f1_macro', ascending=False)
    for vec, row in vec_avg.iterrows():
        print(f"  {vec:20s}: F1={row['f1_macro']:.4f} | Acc={row['accuracy']:.4f} | Prec={row['precision_macro']:.4f} | Rec={row['recall_macro']:.4f}")
    
    # Key insights
    print("\n[KEY INSIGHTS]")
    
    # Linear model advantage
    linear_models = ['log_reg', 'svm']
    nonlinear_models = ['knn', 'random_forest', 'naive_bayes']
    
    linear_f1 = df[df['model'].isin(linear_models)]['f1_macro'].mean()
    nonlinear_f1 = df[df['model'].isin(nonlinear_models)]['f1_macro'].mean()
    
    print(f"  • Linear models (LogReg + SVM) avg F1: {linear_f1:.4f}")
    print(f"  • Non-linear models (KNN + RF) avg F1: {df[df['model']=='knn']['f1_macro'].mean():.4f} (KNN only, severely limited)")
    print(f"  • Naive Bayes avg F1: {df[df['model']=='naive_bayes']['f1_macro'].mean():.4f}")
    
    # TF-IDF vs BoW
    tfidf_f1 = df[df['vectorizer'].str.contains('tfidf')]['f1_macro'].mean()
    bow_f1 = df[df['vectorizer'].str.contains('bow')]['f1_macro'].mean()
    print(f"\n  • TF-IDF average F1: {tfidf_f1:.4f}")
    print(f"  • Bag of Words average F1: {bow_f1:.4f}")
    print(f"  • TF-IDF advantage: +{(tfidf_f1 - bow_f1)*100:.2f}% F1 improvement")
    
    # Unigram vs Bigram
    unigram_f1 = df[df['vectorizer'].str.contains('unigram')]['f1_macro'].mean()
    bigram_f1 = df[df['vectorizer'].str.contains('bigram')]['f1_macro'].mean()
    print(f"\n  • Unigram average F1: {unigram_f1:.4f}")
    print(f"  • Bigram (1-2 gram) average F1: {bigram_f1:.4f}")
    if unigram_f1 > bigram_f1:
        print(f"  • Unigrams better by {(unigram_f1 - bigram_f1)*100:.2f}% F1 on this task")
    
    # Performance range
    print(f"\n  • Best F1 score: {df['f1_macro'].max():.4f}")
    print(f"  • Worst F1 score: {df['f1_macro'].min():.4f}")
    print(f"  • F1 range (best - worst): {(df['f1_macro'].max() - df['f1_macro'].min())*100:.2f}%")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    metrics_path = Path('outputs') / 'metrics.csv'
    stats_path = Path('outputs') / 'dataset_stats.json'
    
    if metrics_path.exists() and stats_path.exists():
        analyze_results(metrics_path, stats_path)
    else:
        print("Error: Run 'python src/train_eval.py' first to generate outputs/")
