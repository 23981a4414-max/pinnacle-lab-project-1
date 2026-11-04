# Fake News Detection (Python, NLP, Machine Learning)

Classifies news text as REAL or FAKE using TF-IDF features and classical ML models.

## Setup
    pip install -r requirements.txt

## Run
    python src/make_sample_data.py      # demo data (synthetic, only for testing the pipeline)
    python src/train.py                 # trains 3 models, saves the best one
    python src/predict.py "Health officials released updated vaccination figures."

## Use a real dataset
Download the "Fake and Real News Dataset" (Fake.csv / True.csv) from Kaggle into data/, then:

    python src/train.py --fake data/Fake.csv --true data/True.csv

## Structure
    src/preprocess.py        text cleaning
    src/train.py             training, comparison, evaluation, saving
    src/predict.py           inference CLI
    models/                  saved pipeline (.joblib)
    reports/                 metrics.json, confusion_matrix.png, top_features.json
