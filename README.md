# SMS Spam Classification using SVM

A machine learning project that classifies SMS text messages as **Spam** or **Ham** (not spam) using TF-IDF feature extraction and a Support Vector Machine (SVM) classifier.

## Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download) (Kaggle / UCI)

- 5,572 SMS messages labeled as `ham` or `spam`
- After removing 403 duplicate messages: 4,825 ham, 747 spam

## Project Structure

- `SMS_Spam_SVM.ipynb` — Jupyter/Colab notebook with the full workflow (EDA, preprocessing, model training, tuning, evaluation)
- `sms_spam_svm.py` — Python script version of the same workflow

## Workflow

1. **Exploratory Data Analysis** — class distribution, message length/word count analysis, word clouds for spam vs. ham messages
2. **Text Preprocessing**
   - Lowercasing
   - Removing URLs, email addresses, and numbers
   - Removing punctuation
   - Removing stopwords (NLTK)
   - Lemmatization (WordNet)
3. **Feature Extraction** — TF-IDF vectorization (unigrams + bigrams, max 5,000 features)
4. **Model** — `LinearSVC` (linear Support Vector Machine) inside a scikit-learn `Pipeline`
5. **Hyperparameter Tuning** — `GridSearchCV` over the SVM's `C` parameter (5-fold CV)
6. **Cross-Validation** — 5-fold CV on the full cleaned dataset to confirm generalization

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.45% |
| Precision | 98.32% |
| Recall | 89.31% |
| F1-score | 93.60% |
| Mean CV Accuracy (5-fold) | 97.95% |

Best hyperparameter: `C = 1`

## Usage

```python
predict_sms("Congratulations! You have won a free iPhone. Click here now.")
# -> "Spam"

predict_sms("Hi, are we meeting tomorrow at 3 PM?")
# -> "Ham"
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
scikit-learn
```

## Author

Shanjida Sultana Jemey
