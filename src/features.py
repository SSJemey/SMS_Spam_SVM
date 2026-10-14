"""Feature engineering: hand‑crafted "meta" features + the combined feature set.

The baseline project only used word TF‑IDF. The enhanced model combines three
views of every message:

1. Word TF‑IDF (unigrams + bigrams) on the enhanced cleaned text
2. Character TF‑IDF (2–5 character n‑grams) on the raw text – catches slang and
   obfuscated words such as "fr33", "txt", "2nite"
3. Meta features – length, digits, capital letters, "!" marks, links, prices…
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .preprocessing import EMAIL_RE, MONEY_RE, PHONE_RE, URL_RE, clean_series

META_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_len",
    "digit_count",
    "digit_ratio",
    "upper_ratio",
    "exclamation_count",
    "special_char_count",
    "has_url",
    "has_email",
    "has_phone",
    "has_money",
]

_SPECIAL_RE = re.compile(r"[£$€%&*#@+]")


def meta_features(text):
    """Return the list of hand‑crafted features for one message."""
    text = str(text)
    words = text.split()
    n_chars = len(text)
    letters = [c for c in text if c.isalpha()]
    digits = sum(c.isdigit() for c in text)
    return [
        np.log1p(n_chars),
        np.log1p(len(words)),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        np.log1p(digits),
        digits / n_chars if n_chars else 0.0,
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        np.log1p(text.count("!")),
        np.log1p(len(_SPECIAL_RE.findall(text))),
        float(bool(URL_RE.search(text))),
        float(bool(EMAIL_RE.search(text))),
        float(bool(PHONE_RE.search(text))),
        float(bool(MONEY_RE.search(text))),
    ]


class MetaFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer that turns raw messages into the meta‑feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([meta_features(t) for t in pd.Series(X)], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(META_FEATURE_NAMES)


def build_features(use_word=True, use_char=True, use_meta=True):
    """Build the combined feature extractor. Flags allow ablation studies."""
    parts = []
    if use_word:
        parts.append((
            "word",
            Pipeline([
                ("clean", FunctionTransformer(clean_series)),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                          sublinear_tf=True)),
            ]),
        ))
    if use_char:
        parts.append((
            "char",
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            max_features=30000, sublinear_tf=True, lowercase=True),
        ))
    if use_meta:
        parts.append((
            "meta",
            Pipeline([("extract", MetaFeatures()), ("scale", StandardScaler())]),
        ))
    return FeatureUnion(parts)


def feature_names(features):
    """Readable names for every column produced by ``build_features``."""
    names = []
    for name, part in features.transformer_list:
        if part == "drop" or part is None:
            continue
        if name == "word":
            names += [f"word: {w}" for w in part.named_steps["tfidf"].get_feature_names_out()]
        elif name == "char":
            names += [f"char: '{c}'" for c in part.get_feature_names_out()]
        elif name == "meta":
            names += [f"meta: {m}" for m in META_FEATURE_NAMES]
    return np.array(names)
