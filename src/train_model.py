"""
train_model.py
---------------
Loads dataset/errors.csv, preprocesses messages, extracts TF-IDF
features, trains and compares two classifiers (Logistic Regression and
Multinomial Naive Bayes; a Linear SVM is included optionally), prints
evaluation metrics, and saves the best-performing model + vectorizer
with joblib.

Run: python src/train_model.py
"""
import os
import sys
import warnings

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_batch  # noqa: E402

warnings.filterwarnings("ignore")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "errors.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["error_message", "error_type"]).drop_duplicates(
        subset=["error_message"]
    )
    return df


def build_vectorizer() -> TfidfVectorizer:
    # unigrams + bigrams: short technical phrases like "no module" or
    # "not defined" are strong signals for these short error messages.
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    print(f"\n{'-' * 50}")
    print(f"Model: {name}")
    print(f"{'-' * 50}")
    print("Model Evaluation")
    print("----------------")
    print(f"Accuracy : {acc:.2f}")
    print(f"Weighted F1 : {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    return acc, f1, preds


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} unique examples across {df['error_type'].nunique()} categories.")

    print("\nPreprocessing error messages...")
    df["processed"] = preprocess_batch(df["error_message"])

    X_text = df["processed"]
    y = df["error_type"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training examples: {len(X_train_text)} | Testing examples: {len(X_test_text)}")

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = {}

    # Model 1: Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train, y_train)
    acc, f1, preds = evaluate("Logistic Regression", log_reg, X_test, y_test)
    results["Logistic Regression"] = (log_reg, acc, f1)
    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())))

    # Model 2: Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    acc, f1, _ = evaluate("Multinomial Naive Bayes", nb, X_test, y_test)
    results["Naive Bayes"] = (nb, acc, f1)

    # Optional Model 3: Linear SVM, calibrated so it also exposes
    # predict_proba (needed for the confidence scores in Phase 12 --
    # a plain LinearSVC only has decision_function, not probabilities).
    min_class_count = y_train.value_counts().min()
    cv_folds = max(2, min(3, min_class_count))
    svm = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE), cv=cv_folds
    )
    svm.fit(X_train, y_train)
    acc, f1, _ = evaluate("Linear SVM (calibrated)", svm, X_test, y_test)
    results["Linear SVM"] = (svm, acc, f1)

    # ---- Comparison table (based on measured results, not a hard-coded pick) ----
    print(f"\n{'=' * 50}")
    print("MODEL COMPARISON")
    print(f"{'=' * 50}")
    print(f"{'Model':<28}{'Accuracy':<12}{'Weighted F1'}")
    print("-" * 50)
    for name, (_, acc, f1) in results.items():
        print(f"{name:<28}{acc * 100:<11.1f}%{f1 * 100:.1f}%")

    best_name = max(results, key=lambda n: results[n][2])  # best weighted F1
    best_model, best_acc, best_f1 = results[best_name]
    print(f"\nBest model (by weighted F1): {best_name} "
          f"(Accuracy={best_acc:.2f}, F1={best_f1:.2f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "error_classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(best_name, os.path.join(MODEL_DIR, "model_name.pkl"))
    # Also keep the full training text (already preprocessed) for the
    # cosine-similarity fallback used when the classifier is uncertain.
    joblib.dump(
        {"texts": X_text.tolist(), "labels": y.tolist(),
         "raw_messages": df["error_message"].tolist()},
        os.path.join(MODEL_DIR, "similarity_corpus.pkl"),
    )

    print(f"\nSaved model      -> {os.path.join(MODEL_DIR, 'error_classifier.pkl')}")
    print(f"Saved vectorizer -> {os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')}")


if __name__ == "__main__":
    main()
