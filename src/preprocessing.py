"""
preprocessing.py
-----------------
Shared NLP preprocessing for the Code Error Explanation Bot.

The SAME `preprocess` function is used by both the training pipeline
(train_model.py) and the prediction pipeline (predictor.py), so that
error messages are always transformed identically.

Design notes:
- Lowercases text but preserves technical tokens (error names, module
  names) rather than stripping them.
- Removes punctuation that carries no classification signal, but keeps
  words attached to it (e.g. "ModuleNotFoundError:" -> "modulenotfounderror").
- Tokenizes, optionally removes stop words, and lemmatizes -- but a
  small set of programming-relevant "stop words" (not, no, is, cannot...)
  is deliberately KEPT because phrases like "not defined" / "no module"
  are useful classification signal for short error messages.
- Falls back to a lightweight, dependency-free implementation if NLTK
  or its data packages are not available (e.g. no internet access),
  so the pipeline never breaks in an offline environment.
"""

import re

# ---------------------------------------------------------------------------
# Try to use NLTK. If it (or its data) isn't available, fall back gracefully.
# ---------------------------------------------------------------------------
_USE_NLTK = False
try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    def _ensure_nltk_data():
        packages = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for path, pkg in packages:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    _ensure_nltk_data()
    _lemmatizer = WordNetLemmatizer()
    _STOPWORDS = set(_nltk_stopwords.words("english"))
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False

# Fallback stopword list (used if NLTK/its corpora are unavailable offline).
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "and", "or", "but",
    "if", "then", "else", "so", "as", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "than", "too", "very", "s", "t", "can", "will", "just", "should",
    "now", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "it", "its", "itself",
    "they", "them", "their", "he", "she", "his", "her", "you", "your",
    "we", "our", "i", "me", "my",
}

# Words that LOOK like stopwords but carry real signal for short error
# messages (e.g. "name is not defined", "no module named"). These are
# NEVER removed, regardless of which stopword source is used.
_KEEP_WORDS = {
    "not", "no", "none", "is", "are", "cannot", "unable", "missing",
    "invalid", "unexpected", "out", "of", "range", "found", "defined",
    "named", "zero", "empty", "expected",
}

# Simple built-in lemmatization rules used only when NLTK's WordNet
# lemmatizer is unavailable. Deliberately conservative (won't touch
# technical identifiers like "numpy" or "modulenotfounderror").
_IRREGULAR = {"indices": "index", "values": "value", "defined": "define",
              "named": "name", "assigned": "assign", "expected": "expect",
              "unmatched": "match"}

# Common technical/package names that must never be touched by the
# fallback stemmer, since naive suffix-stripping would corrupt them
# (e.g. "pandas" -> "panda").
_NO_STEM = {
    "pandas", "numpy", "matplotlib", "requests", "torch", "flask", "cv2",
    "seaborn", "sklearn", "tensorflow", "bs4", "yaml", "class", "address",
    "status",
}


def _fallback_lemmatize(token: str) -> str:
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if token in _NO_STEM:
        return token
    # Only strip a plain trailing plural "s" (conservative: avoids -ed/-ing
    # heuristics, which are too unreliable without real morphology).
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


# Technical/error-domain tokens that must always be preserved verbatim
# and never stemmed/lemmatized away, even though they may look like
# ordinary words after tokenization.
_PROTECTED_SUBSTRINGS = (
    "error", "exception", "traceback",
)


def clean_text(text: str) -> str:
    """Lowercase and strip characters that add noise but keep the
    underlying words (e.g. "'x'" -> "x", "ModuleNotFoundError:" kept
    as a contiguous token)."""
    text = text.strip().lower()
    # Keep letters, digits, underscores, dots (e.g. "numpy.array"), and spaces.
    # Replace everything else with a space so words don't get glued together.
    text = re.sub(r"[^a-z0-9_.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    if _USE_NLTK:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return text.split()


def remove_stopwords(tokens):
    stop = _STOPWORDS if _USE_NLTK else _FALLBACK_STOPWORDS
    return [t for t in tokens if (t not in stop) or (t in _KEEP_WORDS)]


def lemmatize(tokens):
    out = []
    for t in tokens:
        if any(p in t for p in _PROTECTED_SUBSTRINGS):
            out.append(t)  # preserve error-type tokens verbatim
            continue
        if _USE_NLTK:
            out.append(_lemmatizer.lemmatize(t))
        else:
            out.append(_fallback_lemmatize(t))
    return out


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline for a single error message.
    Returns a cleaned, space-joined string ready for TF-IDF vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = [t for t in tokens if t]  # drop empties
    return " ".join(tokens)


def preprocess_batch(texts):
    """Apply preprocess() to an iterable of strings."""
    return [preprocess(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "ModuleNotFoundError: No module named 'pandas'",
        "NameError: name 'total' is not defined",
        "IndexError: list index out of range",
    ]
    for s in samples:
        print(f"{s!r:55s} -> {preprocess(s)!r}")
