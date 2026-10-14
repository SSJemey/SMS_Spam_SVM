"""Model building, saving/loading and explanation helpers."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .features import build_features, feature_names
from .preprocessing import clean_series_original

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_PATH = ROOT / "models" / "spam_svm.joblib"


# ------------------------------------------------------------------- data
def load_data(path=DATA_PATH):
    """Load the Kaggle/UCI SMS Spam Collection and return a clean DataFrame."""
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df = df.drop_duplicates().reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


# ----------------------------------------------------------------- models
def build_baseline():
    """The original project's pipeline (word TF‑IDF + LinearSVC)."""
    return Pipeline([
        ("clean", FunctionTransformer(clean_series_original)),
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("svm", LinearSVC()),
    ])


def build_enhanced(C=1.0, class_weight=None, use_meta=True):
    """Enhanced pipeline: word + char TF‑IDF + meta features → LinearSVC."""
    return Pipeline([
        ("features", build_features(use_meta=use_meta)),
        ("svm", LinearSVC(C=C, class_weight=class_weight, max_iter=10000)),
    ])


def calibrate(pipeline, X, y):
    """Wrap the tuned SVM so it also outputs a spam *probability*.

    LinearSVC only gives a yes/no answer. ``CalibratedClassifierCV`` learns a
    mapping from the SVM's decision score to a probability (Platt scaling).
    """
    svm = pipeline.named_steps["svm"]
    calibrated = Pipeline([
        ("features", pipeline.named_steps["features"]),
        ("svm", CalibratedClassifierCV(
            LinearSVC(C=svm.C, class_weight=svm.class_weight, max_iter=10000),
            method="sigmoid", cv=5)),
    ])
    return calibrated.fit(X, y)


def save_model(model, metadata, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


# ------------------------------------------------------------ explanation
def svm_weights(model):
    """Average linear weights of the SVM(s) inside a (calibrated) pipeline."""
    svm = model.named_steps["svm"]
    if hasattr(svm, "calibrated_classifiers_"):
        return np.mean([c.estimator.coef_[0] for c in svm.calibrated_classifiers_], axis=0)
    return svm.coef_[0]


def top_features(model, n=20):
    """Most spam‑like and most ham‑like features of the trained model."""
    names = feature_names(model.named_steps["features"])
    w = svm_weights(model)
    order = np.argsort(w)
    spam = pd.DataFrame({"feature": names[order[-n:][::-1]], "weight": w[order[-n:][::-1]]})
    ham = pd.DataFrame({"feature": names[order[:n]], "weight": w[order[:n]]})
    return spam, ham


def explain(model, text, n=10):
    """Which features pushed THIS message towards spam or ham?

    For a linear model the contribution of each feature is simply
    ``feature value × weight``, so the explanation is exact.
    """
    feats = model.named_steps["features"]
    x = feats.transform([text])
    x = x.tocsr() if hasattr(x, "tocsr") else x
    w = svm_weights(model)
    names = feature_names(feats)
    idx = x.indices if hasattr(x, "indices") else np.nonzero(x[0])[0]
    vals = x.data if hasattr(x, "data") else x[0][idx]
    contrib = pd.DataFrame({"feature": names[idx], "contribution": vals * w[idx]})
    contrib = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index)
    return contrib.head(n).reset_index(drop=True)


def predict_sms(model, text, threshold=0.5):
    """Return (label, spam_probability) for one message."""
    proba = float(model.predict_proba([text])[0, 1])
    return ("Spam" if proba >= threshold else "Ham"), proba
