"""
predictor.py
------------
Loads the trained model + TF-IDF vectorizer and exposes a single
`predict(error_message)` function used by the explanation engine and
the Streamlit app.

Implements:
  - Phase 11: prediction module (identical preprocessing to training).
  - Phase 12: confidence handling with a configurable threshold.
  - Phase 13: cosine-similarity fallback for uncertain / unusual errors.
"""
import os
import sys

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Phase 12: below this confidence, the result is flagged as uncertain
# and the caller should ask the user for more information / show the
# closest similar known error instead of trusting the raw prediction.
CONFIDENCE_THRESHOLD = 0.45

# Phase 13: below this similarity score, no "closest known error" is
# considered a good enough match to show.
SIMILARITY_THRESHOLD = 0.20


class ErrorPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        model_path = os.path.join(model_dir, "error_classifier.pkl")
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        corpus_path = os.path.join(model_dir, "similarity_corpus.pkl")

        if not (os.path.exists(model_path) and os.path.exists(vec_path)):
            raise FileNotFoundError(
                "Trained model not found. Run `python src/train_model.py` first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        self.has_proba = hasattr(self.model, "predict_proba")

        self.similarity_texts = None
        self.similarity_labels = None
        self.similarity_raw = None
        self.similarity_matrix = None
        if os.path.exists(corpus_path):
            corpus = joblib.load(corpus_path)
            self.similarity_texts = corpus["texts"]
            self.similarity_labels = corpus["labels"]
            self.similarity_raw = corpus["raw_messages"]
            self.similarity_matrix = self.vectorizer.transform(self.similarity_texts)

    def _find_similar(self, processed_text: str, top_k: int = 1):
        """Cosine-similarity fallback: find the closest known error(s)."""
        if self.similarity_matrix is None or not processed_text:
            return []
        query_vec = self.vectorizer.transform([processed_text])
        sims = cosine_similarity(query_vec, self.similarity_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        matches = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < SIMILARITY_THRESHOLD:
                continue
            matches.append({
                "raw_message": self.similarity_raw[idx],
                "error_type": self.similarity_labels[idx],
                "similarity": round(score, 3),
            })
        return matches

    def predict(self, error_message: str) -> dict:
        """
        Predict the error category for a raw error message.

        Returns a dict with:
          error_message, processed_text, predicted_type, confidence,
          is_confident, similar_matches
        """
        if not error_message or not error_message.strip():
            return {
                "error_message": error_message,
                "processed_text": "",
                "predicted_type": None,
                "confidence": 0.0,
                "is_confident": False,
                "similar_matches": [],
                "empty_input": True,
            }

        processed = preprocess(error_message)
        X = self.vectorizer.transform([processed])

        predicted_type = self.model.predict(X)[0]
        confidence = 0.0
        if self.has_proba:
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            # No probability estimate available: fall back to a neutral
            # mid-level confidence so the UI can still function.
            confidence = 0.5

        is_confident = confidence >= CONFIDENCE_THRESHOLD

        similar_matches = []
        if not is_confident:
            similar_matches = self._find_similar(processed, top_k=3)

        return {
            "error_message": error_message,
            "processed_text": processed,
            "predicted_type": predicted_type,
            "confidence": round(confidence, 4),
            "is_confident": is_confident,
            "similar_matches": similar_matches,
            "empty_input": False,
        }


# Module-level singleton so Streamlit doesn't reload the model on every
# interaction (the app additionally wraps this in st.cache_resource).
_predictor_instance = None


def get_predictor() -> ErrorPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ErrorPredictor()
    return _predictor_instance


if __name__ == "__main__":
    p = get_predictor()
    tests = [
        "NameError: name 'x' is not defined",
        "ModuleNotFoundError: No module named 'pandas'",
        "list index out of range",
        "this is some totally unrelated gibberish message",
    ]
    for t in tests:
        result = p.predict(t)
        print(f"{t!r:55s} -> {result['predicted_type']:<20} "
              f"conf={result['confidence']:.2f} confident={result['is_confident']}")
