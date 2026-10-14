"""Streamlit web app for the SMS spam detector.

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import MODEL_PATH, explain, load_model

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

SPAM, HAM = "#eb6834", "#2a78d6"
EXAMPLES = {
    "— pick an example —": "",
    "Prize scam": "Congratulations! You've won a £1000 Tesco gift card. Call 09061743810 now to claim!",
    "Obfuscated spam": "FR33 entry! W1N a £500 prize, txt WIN to 80086 now",
    "Phishing link": "Your account has been suspended. Verify now at www.secure-login-update.com",
    "Normal chat": "Hey, are we still meeting at the library at 3pm tomorrow?",
    "Family message": "I'll be home late, don't wait for dinner",
}


@st.cache_resource
def get_model():
    return load_model()


def pretty(feature):
    kind, value = feature.split(": ", 1)
    return f"{value}  ({'word' if kind == 'word' else 'characters' if kind == 'char' else 'meta'})"


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

model, meta = get_model()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Spam threshold", 0.05, 0.95, 0.50, 0.05,
        help="A message is marked spam when its spam probability is at least this value. "
             "Lower = catch more spam, but more false alarms.")
    st.divider()
    st.subheader("Model")
    tm = meta["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{tm['accuracy']:.1%}")
    c2.metric("F1 (spam)", f"{tm['f1']:.1%}")
    c1.metric("Precision", f"{tm['precision']:.1%}")
    c2.metric("Recall", f"{tm['recall']:.1%}")
    st.caption(
        "Linear SVM on word + character TF‑IDF, calibrated to output probabilities. "
        f"Best C = {meta['best_params']['C']}. Scores are on the {meta['test_size']:,}‑message test set.")

st.title("📩 SMS Spam Detector")
st.caption("Enhanced Support Vector Machine trained on the UCI SMS Spam Collection")

tab_single, tab_batch, tab_about = st.tabs(["Check a message", "Check a CSV file", "How it works"])

# ------------------------------------------------------------ single message
with tab_single:
    choice = st.selectbox("Try an example", list(EXAMPLES))
    text = st.text_area("Message", value=EXAMPLES[choice], height=110,
                        placeholder="Paste or type an SMS here…")
    if st.button("Classify", type="primary") and text.strip():
        proba = float(model.predict_proba([text])[0, 1])
        is_spam = proba >= threshold
        colour = SPAM if is_spam else HAM
        st.markdown(
            f"<div style='padding:18px 22px;border-radius:12px;border-left:8px solid {colour};"
            f"background:{colour}18'><span style='font-size:1.6rem;font-weight:700;color:{colour}'>"
            f"{'🚫 SPAM' if is_spam else '✅ HAM (not spam)'}</span><br>"
            f"Spam probability: <b>{proba:.1%}</b> &nbsp;·&nbsp; threshold {threshold:.0%}</div>",
            unsafe_allow_html=True)
        st.progress(proba)

        st.subheader("Why?")
        exp = explain(model, text, n=12)
        if exp.empty:
            st.info("No known words or patterns found in this message.")
        else:
            exp["feature"] = exp["feature"].map(pretty)
            left, right = st.columns(2)
            with left:
                st.markdown(f"**<span style='color:{SPAM}'>Pushes towards spam</span>**",
                            unsafe_allow_html=True)
                s = exp[exp.contribution > 0]
                if len(s):
                    st.bar_chart(s.set_index("feature")["contribution"], color=SPAM,
                                 horizontal=True)
                else:
                    st.write("—")
            with right:
                st.markdown(f"**<span style='color:{HAM}'>Pushes towards ham</span>**",
                            unsafe_allow_html=True)
                h = exp[exp.contribution < 0].assign(contribution=lambda d: -d.contribution)
                if len(h):
                    st.bar_chart(h.set_index("feature")["contribution"], color=HAM,
                                 horizontal=True)
                else:
                    st.write("—")
            st.caption("Tokens like `phonetoken`, `moneytoken` and `urltoken` stand for a phone "
                       "number, a price and a web link found in the message.")

# ---------------------------------------------------------------- batch CSV
with tab_batch:
    st.write("Upload a CSV file with one message per row.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up, encoding="latin-1")
        col = st.selectbox("Which column holds the messages?", df.columns,
                           index=list(df.columns).index("message") if "message" in df.columns else 0)
        proba = model.predict_proba(df[col].astype(str))[:, 1]
        df["spam_probability"] = proba.round(4)
        df["prediction"] = ["Spam" if p >= threshold else "Ham" for p in proba]
        n_spam = int((df.prediction == "Spam").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", f"{len(df):,}")
        c2.metric("Spam", f"{n_spam:,}")
        c3.metric("Ham", f"{len(df) - n_spam:,}")
        st.dataframe(df, width="stretch", height=360)
        st.download_button("Download results", df.to_csv(index=False).encode(),
                           "sms_predictions.csv", "text/csv")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Pipeline**

1. **Cleaning** – lower‑case, then links → `urltoken`, e‑mails → `emailtoken`,
   prices → `moneytoken`, long numbers → `phonetoken`, other numbers → `numtoken`;
   remove punctuation and stop‑words; lemmatize.
2. **Word TF‑IDF** – unigrams + bigrams of the cleaned text.
3. **Character TF‑IDF** – 2–5 character n‑grams of the raw text (catches *fr33*, *txt*, *2nite*).
4. **Linear SVM** – tuned with 5‑fold grid search, then calibrated to give probabilities.
""")
    figs = Path("results/figures")
    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        comp = pd.DataFrame({"Baseline (original)": m["test_baseline"],
                             "Enhanced": m["test_enhanced"]}).loc[
            ["accuracy", "precision", "recall", "f1", "fn", "fp"]]
        comp.index = ["Accuracy", "Precision", "Recall", "F1", "Missed spam", "False alarms"]
        st.subheader("Baseline vs enhanced (test set)")
        st.dataframe(comp.style.format(lambda v: f"{v:.2%}" if v < 1.0001 and v != int(v) else f"{int(v)}"),
                     width="stretch")
    for name, cap in [("05_confusion_matrices.png", "Confusion matrices"),
                      ("04_ablation_study.png", "What each enhancement adds"),
                      ("08_top_features.png", "Strongest spam and ham signals")]:
        if (figs / name).exists():
            st.image(str(figs / name), caption=cap)
