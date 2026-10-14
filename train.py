"""Train, compare, tune and evaluate the SMS spam models.

Run:  python train.py

Outputs
-------
models/spam_svm.joblib          final calibrated model (used by app.py / predict.py)
results/metrics.json            every number reported in the README
results/*.csv                   comparison tables, misclassified messages
results/figures/*.png           all figures
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src import plots
from src.features import MetaFeatures, META_FEATURE_NAMES, build_features
from src.model import (build_baseline, build_enhanced, calibrate, load_data,
                       save_model, top_features)

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS = Path("results")
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = ["accuracy", "precision", "recall", "f1"]

# Hand‑written messages that were NOT in the dataset. Several use tricks
# spammers use to dodge filters (numbers for letters, odd spacing, $ in words).
ROBUSTNESS_SET = [
    ("FR33 entry! W1N a £500 prize, txt WIN to 80086 now", 1),
    ("Congratulations ur awarded 2 nights holiday. Call 09061743810 to claim", 1),
    ("Your account has been suspended. Verify now at www.secure-login-update.com", 1),
    ("U have been selected for a CA$H reward of $1000. Reply YES to claim", 1),
    ("URGENT! Your mobile number has won a cash bonus. Ring 0800 123 4567", 1),
    ("Get c-h-e-a-p loans today, no credit check!! text LOAN to 61234", 1),
    ("Hey, can you send me the notes from today's class?", 0),
    ("I'll be home late, don't wait for dinner", 0),
    ("Call me when you reach the station, I'm near gate 2", 0),
    ("Your OTP is 482913. Do not share it with anyone.", 0),
    ("Happy birthday!! Have an amazing day, see you tonight", 0),
    ("Can we move the meeting to 3pm tomorrow?", 0),
]


def metrics(y_true, y_pred, score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if score is not None:
        out["roc_auc"] = roc_auc_score(y_true, score)
        out["avg_precision"] = average_precision_score(y_true, score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def log(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ====================================================================== EDA
def eda(df):
    log("1. Exploratory data analysis")
    counts = df["label"].value_counts()
    print(f"Messages after removing duplicates: {len(df)}  ->  {counts.to_dict()}")

    # --- class balance + message length
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bars = ax[0].bar(counts.index, counts.values, color=[plots.HAM, plots.SPAM], width=0.55)
    for b, v in zip(bars, counts.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,} ({v / len(df):.1%})",
                   ha="center", color=plots.INK_2)
    ax[0].set_ylim(0, counts.max() * 1.15)
    ax[0].set_title("Class balance")
    ax[0].set_ylabel("Messages")
    ax[0].grid(axis="x", visible=False)
    lengths = df["message"].str.len()
    for lab, col in [("ham", plots.HAM), ("spam", plots.SPAM)]:
        ax[1].hist(lengths[df.label == lab], bins=np.arange(0, 400, 10), alpha=0.65,
                   color=col, label=lab, density=True)
    ax[1].set_title("Message length (characters)")
    ax[1].set_xlabel("Characters")
    ax[1].set_ylabel("Share of messages")
    ax[1].legend()
    plots.save(fig, FIGS / "01_class_balance_and_length.png")

    # --- meta features: how different are spam and ham?
    meta = pd.DataFrame(MetaFeatures().transform(df["message"]), columns=META_FEATURE_NAMES)
    meta["label"] = df["label"].values
    raw = pd.DataFrame({
        "Characters": df["message"].str.len(),
        "Digits": df["message"].str.count(r"\d"),
        "Upper‑case letters %": 100 * meta["upper_ratio"],
        "'!' marks": df["message"].str.count("!"),
    })
    flags = meta.groupby("label")[["has_url", "has_phone", "has_money", "has_email"]].mean() * 100
    summary = raw.groupby(df["label"].values).mean().round(2)
    summary = pd.concat([summary, flags.rename(columns=lambda c: f"% {c[4:]}").round(2)], axis=1)
    summary.to_csv(RESULTS / "eda_meta_feature_summary.csv")
    print(summary.T.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s = summary[["Characters", "Digits", "Upper‑case letters %", "'!' marks"]].T
    x = np.arange(len(s))
    axes[0].bar(x - 0.2, s["ham"], 0.38, color=plots.HAM, label="ham")
    axes[0].bar(x + 0.2, s["spam"], 0.38, color=plots.SPAM, label="spam")
    axes[0].set_xticks(x, s.index)
    axes[0].set_yscale("log")
    axes[0].set_title("Average per message (log scale)")
    axes[0].legend()
    f = flags.T
    x = np.arange(len(f))
    axes[1].bar(x - 0.2, f["ham"], 0.38, color=plots.HAM, label="ham")
    axes[1].bar(x + 0.2, f["spam"], 0.38, color=plots.SPAM, label="spam")
    axes[1].set_xticks(x, ["URL", "Phone no.", "Price / £", "E‑mail"])
    axes[1].set_ylabel("% of messages")
    axes[1].set_title("Messages containing…")
    axes[1].legend()
    plots.save(fig, FIGS / "02_meta_features.png")
    return summary


# ======================================================== model comparison
def compare_classifiers(X_train, y_train):
    log("2. Comparing classifiers (same word + character TF‑IDF features)")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000),
        "K‑Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "SVM (RBF kernel)": SVC(kernel="rbf", C=10),
        "SVM (linear)": LinearSVC(max_iter=10000),
    }
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("features", build_features(use_meta=False)), ("clf", clf)])
        t = time.time()
        s = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        rows.append({"model": name, **{m: s[f"test_{m}"].mean() for m in SCORING},
                     "f1_std": s["test_f1"].std(), "fit_seconds": s["fit_time"].mean()})
        print(f"{name:26s} F1={rows[-1]['f1']:.4f}  recall={rows[-1]['recall']:.4f}  "
              f"({time.time() - t:.0f}s)")
    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    table.to_csv(RESULTS / "classifier_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    t = table.sort_values("f1")
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["precision"], 0.38, color=plots.HAM, label="Precision")
    ax.barh(y - 0.2, t["recall"], 0.38, color=plots.SPAM, label="Recall")
    for yi, f in zip(y, t["f1"]):
        ax.text(1.002, yi, f"F1 {f:.3f}", va="center", color=plots.INK_2, fontsize=9)
    ax.set_yticks(y, t["model"])
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("5‑fold cross‑validation score (spam class)")
    ax.set_title("Which classifier works best?")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
    ax.grid(axis="y", visible=False)
    plots.save(fig, FIGS / "03_classifier_comparison.png")
    return table


# ============================================================== ablation
def ablation(X_train, y_train):
    log("3. Ablation study – what does each enhancement add?")
    steps = {
        "1. Baseline (original project)": build_baseline(),
        "2. + better cleaning (keep URL/phone/£ tokens)":
            Pipeline([("features", build_features(True, False, False)),
                      ("svm", LinearSVC(max_iter=10000))]),
        "3. + character n‑grams":
            Pipeline([("features", build_features(True, True, False)),
                      ("svm", LinearSVC(max_iter=10000))]),
        "4. + meta features":
            Pipeline([("features", build_features(True, True, True)),
                      ("svm", LinearSVC(max_iter=10000))]),
        "5. + balanced class weights":
            Pipeline([("features", build_features(True, True, True)),
                      ("svm", LinearSVC(class_weight="balanced", max_iter=10000))]),
    }
    rows = []
    for name, pipe in steps.items():
        s = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        rows.append({"step": name, **{m: s[f"test_{m}"].mean() for m in SCORING}})
        print(f"{name:50s} F1={rows[-1]['f1']:.4f}  recall={rows[-1]['recall']:.4f}  "
              f"precision={rows[-1]['precision']:.4f}")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "ablation_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(table))
    for off, m, col, lab in [(-0.27, "precision", plots.HAM, "Precision"),
                             (0, "recall", plots.SPAM, "Recall"),
                             (0.27, "f1", plots.ACCENT, "F1")]:
        ax.bar(x + off, table[m], 0.25, color=col, label=lab)
    ax.set_xticks(x, [s.split(". ", 1)[1].replace(" (", "\n(") for s in table["step"]],
                  fontsize=8.5)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("5‑fold CV score (spam class)")
    ax.set_title("Effect of each enhancement on the linear SVM")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    plots.save(fig, FIGS / "04_ablation_study.png")
    return table


# ================================================================= tuning
def tune(X_train, y_train):
    log("4. Hyper‑parameter tuning (GridSearchCV, optimising F1)")
    pipe = build_enhanced()
    meta_step = pipe.named_steps["features"].transformer_list[2][1]
    grid = {
        "svm__C": [0.1, 0.3, 1, 3, 10],
        "svm__class_weight": [None, "balanced"],
        "features__meta": [meta_step, "drop"],
    }
    search = GridSearchCV(pipe, grid, cv=CV, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_params_
    best_readable = {
        "C": best["svm__C"],
        "class_weight": best["svm__class_weight"],
        "meta_features": best["features__meta"] != "drop",
    }
    print("Best parameters:", best_readable)
    print(f"Best CV F1: {search.best_score_:.4f}")
    res = pd.DataFrame(search.cv_results_)
    res["meta"] = res["param_features__meta"].astype(str).ne("drop")
    res[["param_svm__C", "param_svm__class_weight", "meta", "mean_test_score",
         "std_test_score", "rank_test_score"]].sort_values("rank_test_score").to_csv(
        RESULTS / "grid_search_results.csv", index=False)
    return search.best_estimator_, best_readable, search.best_score_


# ============================================================= evaluation
def evaluate(baseline, final, X_test, y_test):
    log("5. Final evaluation on the held‑out test set")
    b_pred = baseline.predict(X_test)
    b_score = baseline.decision_function(X_test)
    f_proba = final.predict_proba(X_test)[:, 1]
    f_pred = (f_proba >= 0.5).astype(int)
    m_base = metrics(y_test, b_pred, b_score)
    m_final = metrics(y_test, f_pred, f_proba)
    table = pd.DataFrame({"Baseline (original)": m_base, "Enhanced": m_final})
    print(table.round(4).to_string())

    # --- confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (name, pred) in zip(axes, [("Baseline (original)", b_pred), ("Enhanced model", f_pred)]):
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=15,
                    color="white" if v > cm.max() * 0.6 else plots.INK, fontweight="bold")
        ax.set_xticks([0, 1], ["Ham", "Spam"])
        ax.set_yticks([0, 1], ["Ham", "Spam"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        ax.grid(False)
    plots.save(fig, FIGS / "05_confusion_matrices.png")

    # --- ROC and precision–recall
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, s, col in [("Baseline", b_score, plots.GREY), ("Enhanced", f_proba, plots.SPAM)]:
        fpr, tpr, _ = roc_curve(y_test, s)
        axes[0].plot(fpr, tpr, color=col, label=f"{name} (AUC {auc(fpr, tpr):.4f})")
        p, r, _ = precision_recall_curve(y_test, s)
        axes[1].plot(r, p, color=col, label=f"{name} (AP {average_precision_score(y_test, s):.4f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=plots.GREY, lw=1)
    axes[0].set_xlim(-0.01, 0.3)
    axes[0].set_ylim(0.7, 1.005)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curve (zoomed to top‑left)")
    axes[1].set_xlim(0.6, 1.005)
    axes[1].set_ylim(0.6, 1.005)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (zoomed)")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    plots.save(fig, FIGS / "06_roc_pr_curves.png")

    # --- decision threshold trade‑off
    th = np.linspace(0.05, 0.95, 91)
    prec = [precision_score(y_test, f_proba >= t, zero_division=1) for t in th]
    rec = [recall_score(y_test, f_proba >= t) for t in th]
    f1s = [f1_score(y_test, f_proba >= t) for t in th]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(th, prec, color=plots.HAM, label="Precision")
    ax.plot(th, rec, color=plots.SPAM, label="Recall")
    ax.plot(th, f1s, color=plots.ACCENT, label="F1")
    ax.axvline(0.5, ls="--", color=plots.GREY, lw=1)
    ax.text(0.51, min(rec) + 0.005, "default 0.5", color=plots.INK_2, fontsize=9)
    ax.set(xlabel="Spam probability threshold", ylabel="Score",
           title="Choosing the decision threshold")
    ax.legend(loc="lower left")
    plots.save(fig, FIGS / "07_threshold_tradeoff.png")
    threshold_table = pd.DataFrame({"threshold": th, "precision": prec, "recall": rec, "f1": f1s})
    threshold_table.round(4).to_csv(RESULTS / "threshold_tradeoff.csv", index=False)

    # --- error analysis
    errors = pd.DataFrame({"message": X_test.values, "actual": y_test.values,
                           "predicted": f_pred, "spam_probability": f_proba.round(4)})
    errors = errors[errors.actual != errors.predicted].copy()
    errors["error_type"] = np.where(errors.actual == 1, "missed spam (false negative)",
                                    "ham flagged as spam (false positive)")
    errors.sort_values("spam_probability").to_csv(RESULTS / "misclassified_messages.csv", index=False)
    print(f"\nMisclassified test messages: {len(errors)} (saved to results/misclassified_messages.csv)")
    return m_base, m_final


def plot_top_features(final):
    spam, ham = top_features(final, n=15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, d, col, title in [(axes[0], spam, plots.SPAM, "Strongest SPAM signals"),
                              (axes[1], ham, plots.HAM, "Strongest HAM signals")]:
        d = d.iloc[::-1]
        ax.barh(d["feature"], d["weight"].abs(), color=col)
        ax.set_title(title)
        ax.set_xlabel("|SVM weight|")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    plots.save(fig, FIGS / "08_top_features.png")
    pd.concat([spam.assign(direction="spam"), ham.assign(direction="ham")]).to_csv(
        RESULTS / "top_features.csv", index=False)


def robustness(baseline, final):
    log("6. Robustness test on new, hand‑written messages")
    msgs = [m for m, _ in ROBUSTNESS_SET]
    truth = np.array([t for _, t in ROBUSTNESS_SET])
    b = baseline.predict(msgs)
    p = final.predict_proba(msgs)[:, 1]
    table = pd.DataFrame({
        "message": msgs,
        "actual": np.where(truth == 1, "Spam", "Ham"),
        "baseline": np.where(b == 1, "Spam", "Ham"),
        "enhanced": np.where(p >= 0.5, "Spam", "Ham"),
        "enhanced_spam_prob": p.round(3),
    })
    table.to_csv(RESULTS / "robustness_test.csv", index=False)
    print(table.to_string(index=False))
    acc_b = (b == truth).mean()
    acc_f = ((p >= 0.5).astype(int) == truth).mean()
    print(f"\nBaseline correct: {int(acc_b * len(truth))}/{len(truth)}   "
          f"Enhanced correct: {int(acc_f * len(truth))}/{len(truth)}")
    return {"baseline_correct": int((b == truth).sum()),
            "enhanced_correct": int(((p >= 0.5).astype(int) == truth).sum()),
            "total": len(truth)}


# =================================================================== main
def main():
    plots.apply_style()
    df = load_data()
    eda_summary = eda(df)

    X, y = df["message"], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")

    clf_table = compare_classifiers(X_train, y_train)
    abl_table = ablation(X_train, y_train)
    best_pipe, best_params, best_cv_f1 = tune(X_train, y_train)

    baseline = build_baseline().fit(X_train, y_train)
    final = calibrate(best_pipe, X_train, y_train)

    m_base, m_final = evaluate(baseline, final, X_test, y_test)
    plot_top_features(final)
    robust = robustness(baseline, final)

    log("7. 5‑fold cross‑validation on the full dataset")
    cv_base = cross_validate(build_baseline(), X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    cv_final = cross_validate(clone(best_pipe), X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    cv_summary = {
        name: {m: float(r[f"test_{m}"].mean()) for m in SCORING}
        for name, r in [("baseline", cv_base), ("enhanced", cv_final)]
    }
    print(pd.DataFrame(cv_summary).round(4).to_string())

    metadata = {
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "test_metrics": m_final,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    save_model(final, metadata)

    all_metrics = {
        "dataset": {"messages": len(df), "ham": int((y == 0).sum()), "spam": int((y == 1).sum()),
                    "train": len(X_train), "test": len(X_test)},
        "test_baseline": m_base,
        "test_enhanced": m_final,
        "cv_full_dataset": cv_summary,
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "robustness": robust,
        "classifier_comparison": clf_table.to_dict(orient="records"),
        "ablation": abl_table.to_dict(orient="records"),
    }
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    print("\nSaved model to models/spam_svm.joblib and results to results/")


if __name__ == "__main__":
    main()
