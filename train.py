"""Train and evaluate fake-news classifiers.

Usage
-----
Demo data:        python src/train.py
Kaggle dataset:   python src/train.py --fake data/Fake.csv --true data/True.csv
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
from preprocess import clean_text


def load_data(args):
    if args.fake and args.true:
        fake = pd.read_csv(args.fake); fake["label"] = 0
        true = pd.read_csv(args.true); true["label"] = 1
        df = pd.concat([fake, true], ignore_index=True)
        if "title" in df:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df = df[["text", "label"]]
    else:
        df = pd.read_csv(args.data)
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    df["clean"] = df["text"].map(clean_text)
    return df


def build(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  max_df=0.9, min_df=2, sublinear_tf=True)),
        ("clf", model),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_news.csv")
    ap.add_argument("--fake"); ap.add_argument("--true")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"], test_size=args.test_size, stratify=df["label"], random_state=42)
    print(f"Rows: {len(df)} | train {len(X_tr)} | test {len(X_te)}")

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }
    results, best_name, best_f1, best_pipe = {}, None, -1, None
    for name, model in candidates.items():
        pipe = build(model).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary")
        results[name] = dict(accuracy=accuracy_score(y_te, pred), precision=p, recall=r, f1=f1)
        print(f"{name:20s} acc={results[name]['accuracy']:.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_f1, best_pipe = name, f1, pipe

    joblib.dump(best_pipe, "models/fake_news_model.joblib")
    with open("reports/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    cm = confusion_matrix(y_te, best_pipe.predict(X_te))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout(); fig.savefig("reports/confusion_matrix.png", dpi=200)

    clf = best_pipe.named_steps["clf"]
    coef = None
    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):   # Naive Bayes: log-odds of REAL vs FAKE
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    if coef is not None:
        vocab = np.array(best_pipe.named_steps["tfidf"].get_feature_names_out())
        with open("reports/top_features.json", "w") as f:
            json.dump({"fake": vocab[np.argsort(coef)[:10]].tolist(),
                       "real": vocab[np.argsort(coef)[-10:]].tolist()}, f, indent=2)
    print(f"Best model: {best_name} -> saved to models/fake_news_model.joblib")


if __name__ == "__main__":
    main()
