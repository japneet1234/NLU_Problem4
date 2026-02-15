# Sports vs Politics Classification Results

**Dataset:** AG News (60,000 documents, 30,000 sports + 30,000 politics)  
**Train/Test Split:** 70% train (42,000) / 30% test (18,000)  
**Models:** 5 (Logistic Regression, SVM, KNN, Naive Bayes, Random Forest)  
**Vectorizers:** 4 (BoW unigram, BoW bigram, TF-IDF unigram, TF-IDF bigram)

## Performance Summary

| Rank | Vectorizer | Model | Accuracy | Precision | Recall | F1 Score |
|------|------------|-------|----------|-----------|--------|----------|
| 1 | TF-IDF bigram | SVM | 0.9774 | 0.9776 | 0.9774 | **0.9774** |
| 2 | TF-IDF unigram | SVM | 0.9762 | 0.9763 | 0.9762 | **0.9762** |
| 3 | BoW bigram | Logistic Reg | 0.9751 | 0.9752 | 0.9751 | **0.9751** |
| 4 | BoW bigram | Naive Bayes | 0.9741 | 0.9745 | 0.9741 | **0.9741** |
| 5 | TF-IDF unigram | Logistic Reg | 0.9736 | 0.9738 | 0.9736 | **0.9736** |
| 6 | TF-IDF bigram | Naive Bayes | 0.9730 | 0.9734 | 0.9730 | **0.9730** |
| 7 | BoW unigram | Naive Bayes | 0.9729 | 0.9732 | 0.9729 | **0.9729** |
| 8 | BoW bigram | SVM | 0.9727 | 0.9727 | 0.9727 | **0.9727** |
| 9 | TF-IDF unigram | Naive Bayes | 0.9724 | 0.9727 | 0.9724 | **0.9724** |
| 10 | TF-IDF bigram | Logistic Reg | 0.9723 | 0.9726 | 0.9723 | **0.9723** |
| 11 | BoW unigram | Logistic Reg | 0.9716 | 0.9717 | 0.9716 | **0.9716** |
| 12 | TF-IDF unigram | KNN | 0.9678 | 0.9680 | 0.9678 | **0.9678** |
| 13 | TF-IDF bigram | KNN | 0.9677 | 0.9678 | 0.9677 | **0.9677** |
| 14 | BoW bigram | Random Forest | 0.9643 | 0.9648 | 0.9643 | **0.9643** |
| 15 | BoW unigram | Random Forest | 0.9628 | 0.9632 | 0.9628 | **0.9628** |
| 16 | TF-IDF unigram | Random Forest | 0.9622 | 0.9626 | 0.9622 | **0.9622** |
| 17 | TF-IDF bigram | Random Forest | 0.9622 | 0.9628 | 0.9622 | **0.9622** |
| 18 | BoW unigram | SVM | 0.9588 | 0.9588 | 0.9588 | **0.9588** |
| 19 | BoW unigram | KNN | 0.7994 | 0.8013 | 0.7994 | **0.7991** |
| 20 | BoW bigram | KNN | 0.6527 | 0.6728 | 0.6527 | **0.6423** |

## Key Findings

### Best Configuration
- **Winner:** TF-IDF bigram + SVM (F1 = 0.9774)
- Top 3 configurations all exceed 97.5% F1 score
- Linear models (Logistic Regression & SVM) dominate the leaderboard

### Model Comparison (Average F1)
1. **Logistic Regression:** 0.9731 (rank 1-2)
2. **Naive Bayes:** 0.9731 (rank 1-2)
3. **SVM:** 0.9713 (rank 3)
4. **Random Forest:** 0.9629 (rank 4)
5. **KNN:** 0.8442 (rank 5) - struggles significantly

### Vectorizer Comparison (Average F1)
1. **TF-IDF bigram:** 0.9705 (+5.11% over BoW)
2. **TF-IDF unigram:** 0.9704
3. **BoW unigram:** 0.9330
4. **BoW bigram:** 0.9057

### Insights
- **TF-IDF advantage:** +5.11% F1 over Bag of Words
- **Unigrams slightly better:** +1.36% F1 over bigrams
- **KNN failure:** Suffers from curse of dimensionality (worst F1: 0.6423)
- **Linear models excel:** Logistic Regression & SVM best for sparse text data

## Dataset Note

⚠️ **BBC → AG News Transition:** We initially used the BBC News dataset but found it produced unrealistic 100% accuracy across multiple models due to highly distinct topical vocabularies. We switched to AG News (60K documents) for more realistic and challenging evaluation with vocabulary overlap between sports and politics categories.
