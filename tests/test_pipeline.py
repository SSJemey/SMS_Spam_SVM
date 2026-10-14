"""Basic tests. Run with:  pytest -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import META_FEATURE_NAMES, MetaFeatures  # noqa: E402
from src.model import MODEL_PATH, explain, load_model, predict_sms  # noqa: E402
from src.preprocessing import clean_text, clean_text_v2  # noqa: E402


def test_original_cleaner_removes_numbers_and_links():
    out = clean_text("Call 09061743810 or visit www.win.com NOW!")
    assert "09061743810" not in out
    assert "www" not in out
    assert out == out.lower()


def test_enhanced_cleaner_keeps_spam_signals_as_tokens():
    out = clean_text_v2("Win £500! Call 09061743810 or visit www.win.com, email a@b.com")
    for token in ["moneytoken", "phonetoken", "urltoken", "emailtoken"]:
        assert token in out


def test_meta_features_shape_and_flags():
    X = MetaFeatures().transform(["Hello there", "WIN £100 now!!! call 09061743810"])
    assert X.shape == (2, len(META_FEATURE_NAMES))
    names = list(META_FEATURE_NAMES)
    assert X[0, names.index("has_phone")] == 0
    assert X[1, names.index("has_phone")] == 1
    assert X[1, names.index("has_money")] == 1


needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train.py first")


@needs_model
def test_model_predicts_obvious_cases():
    model, _ = load_model()
    assert predict_sms(model, "Congratulations! You won a £1000 prize. Call 09061743810 now")[0] == "Spam"
    assert predict_sms(model, "Are we still meeting for lunch tomorrow?")[0] == "Ham"


@needs_model
def test_explanation_returns_features():
    model, _ = load_model()
    exp = explain(model, "WIN a free prize, call 09061743810", n=5)
    assert len(exp) == 5
    assert exp["contribution"].iloc[0] > 0
