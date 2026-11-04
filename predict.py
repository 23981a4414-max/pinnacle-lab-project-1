"""Classify a news headline/article.

Usage: python src/predict.py "Your news text here"
"""
import math
import sys
import joblib
from preprocess import clean_text

LABELS = {0: "FAKE", 1: "REAL"}


def predict(text: str, model_path: str = "models/fake_news_model.joblib"):
    model = joblib.load(model_path)
    x = [clean_text(text)]
    label = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(x)[0][label])
    else:  # LinearSVC has no probabilities: squash the decision score into a rough 0-1 value
        conf = 1 / (1 + math.exp(-abs(float(model.decision_function(x)[0]))))
    return LABELS[label], conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/predict.py "news text"')
    lab, conf = predict(" ".join(sys.argv[1:]))
    print(f"Prediction: {lab} (confidence ~{conf:.0%})")
