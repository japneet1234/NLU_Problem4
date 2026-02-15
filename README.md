# Sports vs Politics Text Classification

A machine learning project that classifies news articles as either Sports or Politics. Built and tested with 5 different models and multiple text representation techniques.

## What's this about?

This is my NLU Problem 4. The goal was to build a text classifier that can tell sports news from politics news. I tested different machine learning models and feature extraction methods to see which combination works best.

## Quick Results

The best setup I found was **TF-IDF + SVM** which got **97.74% accuracy**. Pretty solid!

Top performers:
- TF-IDF (bigram) + SVM: 97.74%
- TF-IDF (unigram) + SVM: 97.62%
- BoW (bigram) + Logistic Regression: 97.51%

## What I tested

**Models:** Logistic Regression, SVM, KNN, Naive Bayes, Random Forest

**Text features:** Bag of Words (unigram/bigram) and TF-IDF (unigram/bigram)

**Dataset:** 60,000 news articles from AG News (30,000 Sports + 30,000 Politics)


## Key findings

1. **Linear models are best** - Both Logistic Regression and SVM performed really well
2. **TF-IDF is much better than Bag of Words** - About 5% improvement
3. **KNN doesn't work well here** - High-dimensional text data isn't its friend
4. **BBC dataset had a problem** - Too easy (100% accuracy), so I switched to AG News

## Links

- View results: https://japneet1234.github.io/NLU_Problem4/
- See the code: https://github.com/japneet1234/NLU_Problem4
- Full report: https://github.com/japneet1234/NLU_Problem4/blob/main/B23CS1022_report.tex

## Author
Japneet Singh (B23CS1022)
