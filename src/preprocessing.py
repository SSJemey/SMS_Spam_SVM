"""Text cleaning functions.

Two cleaners live here:

* ``clean_text`` – the ORIGINAL cleaner from the baseline notebook. It deletes
  URLs, e‑mails, numbers and punctuation.
* ``clean_text_v2`` – the ENHANCED cleaner. Instead of deleting the things that
  make spam look like spam (links, phone numbers, prices, numbers) it replaces
  them with placeholder tokens such as ``urltoken`` or ``phonetoken`` so the
  model can still learn from them.
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
    """Download the NLTK corpora on first use (stop‑words + WordNet)."""
    for resource, package in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------- patterns
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|co\.uk|biz|info)\S*)", re.I)
EMAIL_RE = re.compile(r"\S+@\S+")
MONEY_RE = re.compile(r"(£|\$|€)\s?\d+(?:[.,]\d+)?|\b\d+(?:\.\d+)?\s?(?:p|pence|gbp|usd)\b", re.I)
PHONE_RE = re.compile(r"\b\d{5,}\b")          # long digit runs: phone / short codes
NUMBER_RE = re.compile(r"\d+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation + "£€")


def _tokens(text):
    """Remove punctuation, stopwords and lemmatize."""
    text = text.translate(PUNCT_TABLE)
    return [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]


def clean_text(text):
    """Original cleaner (baseline). Kept unchanged for a fair comparison."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def clean_text_v2(text):
    """Enhanced cleaner: keeps spam signals as placeholder tokens."""
    text = text.lower()
    text = EMAIL_RE.sub(" emailtoken ", text)   # before URLs: "a@b.com" is an e-mail
    text = URL_RE.sub(" urltoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    return " ".join(_tokens(text))


def clean_series(texts):
    """Apply ``clean_text_v2`` to a list / Series (used inside the sklearn pipeline)."""
    return [clean_text_v2(t) for t in pd.Series(texts).astype(str)]


def clean_series_original(texts):
    """Apply the original ``clean_text`` to a list / Series."""
    return [clean_text(t) for t in pd.Series(texts).astype(str)]
