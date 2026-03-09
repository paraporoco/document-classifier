"""
vectorizer.py — TF-IDF feature extraction.

Fit on training data only (no leakage into val/unlabelled).
Persists to disk so inference reuses the same vocabulary.

Design choices:
  max_features = 5000   — caps vocabulary; sufficient for this signal vocabulary
  ngram_range  = (1,2)  — bigrams capture multi-word signals ("internal use only")
  sublinear_tf = True   — log-normalise term frequencies; reduces dominance of
                          repeated terms
"""

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_PATH = Path(__file__).parent.parent / "artefacts" / "vectorizer.pkl"


def fit(texts: list[str]) -> TfidfVectorizer:
    """Fit and return a TF-IDF vectorizer on `texts`."""
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    vec.fit(texts)
    return vec


def transform(vec: TfidfVectorizer, texts: list[str]) -> np.ndarray:
    """Transform `texts` using a fitted vectorizer. Returns dense array."""
    return vec.transform(texts).toarray().astype(np.float32)


def save(vec: TfidfVectorizer, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vec, f)
    print(f"Vectorizer saved → {path}")


def load(path: Path = DEFAULT_PATH) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generator import generate

    labelled, _ = generate()
    texts = [s.text for s in labelled]
    vec = fit(texts)
    X = transform(vec, texts)
    print(f"Vocabulary size : {len(vec.vocabulary_)}")
    print(f"Feature matrix  : {X.shape}")
    save(vec)
