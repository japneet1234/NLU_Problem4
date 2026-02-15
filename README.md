# 🏆 Sports vs Politics Text Classification

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/japneet1234/NLU_Problem4)

A comprehensive machine learning study comparing 5 classical models with 4 feature representations for binary text classification on the AG News dataset.

[View Results](https://japneet1234.github.io/NLU-Problem4/) • [Read Report](report/report.tex) • [View Metrics](outputs/metrics.md)

</div>

---

## 📋 Project Overview

This project implements and evaluates a **binary text classification system** that distinguishes between **Sports** and **Politics** news articles. The work compares:

- **5 Machine Learning Models**: Logistic Regression, SVM, KNN, Naive Bayes, Random Forest
- **4 Feature Representations**: BoW unigram, BoW bigram, TF-IDF unigram, TF-IDF bigram
- **60,000 Documents** from AG News dataset with perfect 50-50 class balance

### 🎯 Best Performance
- **Model**: TF-IDF (bigram) + SVM
- **F1 Score**: **97.74%**
- **Accuracy**: **97.74%**

---

## 📊 Key Results

| Rank | Vectorizer | Model | F1 Score |
|------|------------|-------|----------|
| 🥇 1 | TF-IDF (1-2 gram) | SVM | **0.9774** |
| 🥈 2 | TF-IDF (1-gram) | SVM | **0.9762** |
| 🥉 3 | BoW (1-2 gram) | Logistic Reg | **0.9751** |
| 4 | BoW (1-2 gram) | Naive Bayes | **0.9741** |
| 5 | TF-IDF (1-gram) | Logistic Reg | **0.9736** |

### 💡 Key Findings

- **Linear Models Excel**: Logistic Regression & SVM avg F1 = 0.9731
- **TF-IDF Advantage**: +5.11% F1 over Bag of Words
- **KNN Struggles**: Curse of dimensionality (F1 = 0.64-0.80)
- **Multiple Strong Configs**: Top 5 all exceed 97.3% F1

---

## 🛠️ Project Structure

```
NLU_Problem4/
├── src/
│   ├── train_eval.py          # Main training & evaluation script
│   └── analyze_results.py     # Results analysis & insights
├── report/
│   └── report.tex             # Detailed LaTeX academic report (5+ pages)
├── docs/
│   └── index.html             # Interactive GitHub Pages results page
├── outputs/
│   ├── metrics.csv            # Performance metrics for all 20 configs
│   ├── metrics.md             # Formatted results table
│   └── dataset_stats.json     # Dataset statistics
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/japneet1234/NLU_Problem4.git
cd NLU_Problem4

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Training & Evaluation

```bash
# Train on AG News dataset
python src/train_eval.py --data-path archive/train.csv

# Analyze results
python src/analyze_results.py
```

**Note**: The AG News dataset is automatically downloaded during training if not present.

---

## 📈 Dataset Information

**AG News** (60,000 documents, 30,000 per class):
- **Sports Class**: Coverage of matches, leagues, athletes, tournaments
- **Politics Class**: World news mapped to politics (elections, policy, international relations)
- **Train/Test Split**: 70% train (42,000) / 30% test (18,000)
- **Advantage**: More realistic challenge with vocabulary overlap vs. BBC dataset

### 🔄 Dataset Evolution

Initially trained on **BBC News dataset** (928 docs) but achieved unrealistic 100% accuracy due to highly distinct topical vocabularies. Switched to **AG News** for more challenging and realistic evaluation.

---

## 🔬 Machine Learning Models

### Linear Models
- **Logistic Regression**: Simple, interpretable, effective on sparse text
- **Support Vector Machine (SVM)**: Maximum margin classifier, excellent generalization

### Probabilistic Models
- **Multinomial Naive Bayes**: Fast training, strong independence assumptions

### Instance-Based & Tree-Based
- **K-Nearest Neighbors**: Struggles with high-dimensional sparse data (curse of dimensionality)
- **Random Forest**: Ensemble method, limited effectiveness on sparse features

---

## 📝 Feature Representations

### Bag of Words (BoW)
Raw word counts with optional bigrams. Simple but effective baseline.

### TF-IDF
Term Frequency-Inverse Document Frequency weighting:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

Downweights common words, emphasizes domain-specific keywords. **+5.1% F1 advantage over BoW**.

---

## 📚 Detailed Report

A comprehensive **5+ page LaTeX academic report** is available in [`report/report.tex`](report/report.tex) covering:

- **Introduction**: Problem formulation and motivation
- **Data Collection**: Dataset selection and BBC→AG News transition
- **Dataset Description**: Statistics, class distribution, characteristics
- **Feature Representations**: Mathematical formulations of BoW and TF-IDF
- **Models**: Architecture and training details for all 5 classifiers
- **Experimental Setup**: Data split, evaluation protocol, metrics
- **Results**: Quantitative comparison with ranking tables
- **Discussion**: Model-by-model analysis, key observations
- **Limitations**: Dataset ambiguity, vocabulary overlap, single split, etc.
- **Conclusion & Future Work**: Synthesis and directions for improvement

---

## 🌐 Interactive Results

Visit the **GitHub Pages project page** for an interactive visualization:
👉 [https://japneet1234.github.io/NLU-Problem4/](https://japneet1234.github.io/NLU-Problem4/)

Features:
- 📊 Live statistics dashboard
- 🏆 Leaderboard of all configurations
- 💡 Key findings & insights
- 📈 Model & vectorizer performance rankings

---

## 📂 Output Files

### `outputs/metrics.csv`
Raw performance metrics for all 20 model-vectorizer combinations:
```csv
vectorizer,model,accuracy,precision_macro,recall_macro,f1_macro
tfidf_bigram,svm,0.9774,0.9776,0.9774,0.9774
...
```

### `outputs/metrics.md`
Formatted results table with rankings and insights

### `outputs/dataset_stats.json`
Dataset statistics (total docs, class counts, train/test sizes)

---

## 🔑 Key Insights

1. **Linear Models Dominate**: Both Logistic Regression and SVM achieve F1 ≈ 0.973, showing linear separability in high-dimensional text spaces

2. **TF-IDF is Critical**: Inverse document frequency weighting provides +5.11% F1 improvement, vital for distinguishing topical terms

3. **Unigrams > Bigrams**: Individual words (+1.36% F1) more informative than word pairs for this task

4. **KNN Fails Spectacularly**: Curse of dimensionality in sparse spaces; worst config (BoW bigram + KNN) achieves only 64.23% F1

5. **Dataset Matters**: BBC dataset too clean (100% accuracy) → AG News provides realistic benchmark with overlapping vocabulary

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Single train-test split (k-fold cross-validation recommended)
- No hyperparameter tuning (systematic grid search could improve performance)
- Limited text preprocessing (stemming, lemmatization not applied)
- Binary classification only (extensible to multi-class)
- English-only (multilingual support as future direction)

### Future Improvements
- Dense word embeddings (Word2Vec, FastText, transformers)
- Advanced hyperparameter tuning & cross-validation
- Larger & more diverse datasets
- Multi-class extension to all AG News categories
- Attention mechanisms & LSTM-based approaches

---

## 📦 Dependencies

```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
```

See `requirements.txt` for complete list.

---

## 👤 Author

**Japneet Singh**  
Roll Number: B23CS1022

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- 📊 [Interactive Results Page](https://japneet1234.github.io/NLU-Problem4/)
- 📖 [LaTeX Report](report/report.tex)
- 📋 [Results Metrics](outputs/metrics.md)
- 🐙 [GitHub Repository](https://github.com/japneet1234/NLU_Problem4)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>
