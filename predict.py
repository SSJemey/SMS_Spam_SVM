"""Classify SMS messages from the command line.

Examples
--------
python predict.py "Congratulations! You have won a free iPhone. Call 09061743810"
python predict.py --explain "WIN a £500 prize, txt WIN to 80086"
python predict.py --file my_messages.csv --column message --out predictions.csv
"""

import argparse
import sys

import pandas as pd

from src.model import MODEL_PATH, explain, load_model, predict_sms


def main():
    parser = argparse.ArgumentParser(description="SMS spam detector (enhanced SVM)")
    parser.add_argument("message", nargs="*", help="message(s) to classify")
    parser.add_argument("--file", help="CSV file with messages to classify in bulk")
    parser.add_argument("--column", default="message", help="column holding the text")
    parser.add_argument("--out", default="predictions.csv", help="where to write bulk results")
    parser.add_argument("--threshold", type=float, default=0.5, help="spam probability cut‑off")
    parser.add_argument("--explain", action="store_true", help="show the words behind the decision")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run `python train.py` first.")
    model, _ = load_model()

    if args.file:
        df = pd.read_csv(args.file, encoding="latin-1")
        proba = model.predict_proba(df[args.column].astype(str))[:, 1]
        df["spam_probability"] = proba.round(4)
        df["prediction"] = ["Spam" if p >= args.threshold else "Ham" for p in proba]
        df.to_csv(args.out, index=False)
        print(df["prediction"].value_counts().to_string())
        print(f"Saved {len(df)} predictions to {args.out}")
        return

    if not args.message:
        parser.print_help()
        return

    for text in args.message:
        label, proba = predict_sms(model, text, args.threshold)
        print(f"[{label:4s}] {proba:6.1%}  {text}")
        if args.explain:
            for _, row in explain(model, text, n=8).iterrows():
                arrow = "→ spam" if row.contribution > 0 else "→ ham "
                print(f"          {arrow} {row.contribution:+.3f}  {row.feature}")


if __name__ == "__main__":
    main()
