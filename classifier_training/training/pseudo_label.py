"""
pseudo_label.py — Semi-supervised self-training via pseudo-labelling.

Algorithm:
    Start from the supervised checkpoint (model + vectorizer + temperature).

    Repeat for N iterations:
        1. Run calibrated model over unlabelled pool
        2. Accept predictions above the Kth percentile of confidence scores
           (percentile-based threshold: adapts to actual confidence scale)
        3. Append accepted (text, predicted_label) pairs to training set
        4. Retrain model on the expanded training set (warm_start=True)
        5. Refit temperature on the new validation set
        6. Lower percentile cutoff each round — harder samples absorbed later

    Stop when no new samples pass the threshold, or max iterations reached.

Why percentile-based threshold:
    A fixed threshold (e.g. 0.85) fails at high noise because the model's
    confidence scores collapse below 0.85 for all samples. A percentile-based
    threshold guarantees that the top K% of predictions are always accepted,
    regardless of the absolute confidence scale.
"""

import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import CLASSES, generate
from features.vectorizer import load as load_vec, transform
from features.calibration import (
    calibrated_predict, fit_temperature, save_temperature, load_temperature
)
from model.net import build_network

ARTEFACTS       = Path(__file__).parent.parent / "artefacts"
MODEL_PATH      = ARTEFACTS / "model.pkl"
SSL_MODEL_PATH  = ARTEFACTS / "model_ssl.pkl"

INITIAL_PERCENTILE = 80   # first iteration: accept top 20% most confident
PERCENTILE_DECAY   = 10   # lower bar each iteration by this many percentile points
MAX_ITERATIONS     = 5
MIN_PERCENTILE     = 40   # never go below 40th percentile


def run(seed: int = 42, verbose: bool = True):
    # ── Load base supervised checkpoint ────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    vec = load_vec()
    T   = load_temperature()

    # ── Load data ──────────────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    labelled, unlabelled = generate(seed=seed)

    texts  = [s.text      for s in labelled]
    labels = [s.label_idx for s in labelled]

    _, X_texts_val, _, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )

    train_texts  = list(texts)
    train_labels = list(labels)

    unlab_texts  = [s.text for s in unlabelled]
    X_unlab      = transform(vec, unlab_texts)
    remaining    = np.ones(len(unlab_texts), dtype=bool)

    percentile = INITIAL_PERCENTILE

    if verbose:
        print(f"Starting semi-supervised self-training  (T={T:.3f})")
        print(f"Unlabelled pool    : {len(unlab_texts)} samples")
        print(f"Initial percentile : {percentile}th\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        if not remaining.any():
            print("All unlabelled samples absorbed. Stopping.")
            break

        # ── Step 1: calibrated prediction over remaining unlabelled ────────────
        X_rem = X_unlab[remaining]
        predicted, confidence = calibrated_predict(model, X_rem, T)

        # ── Step 2: percentile-based threshold ─────────────────────────────────
        threshold    = np.percentile(confidence, percentile)
        accepted_rel = confidence >= threshold
        n_accepted   = accepted_rel.sum()

        if verbose:
            print(f"Iteration {iteration}  |  "
                  f"percentile={percentile}th  |  "
                  f"threshold={threshold:.3f}  |  "
                  f"accepted={n_accepted}/{remaining.sum()}")

        # ── Step 3: add pseudo-labelled samples ────────────────────────────────
        remaining_idx    = np.where(remaining)[0]
        accepted_global  = remaining_idx[accepted_rel]

        new_texts  = [unlab_texts[i]      for i in accepted_global]
        new_labels = predicted[accepted_rel].tolist()

        if verbose:
            dist = Counter(CLASSES[l] for l in new_labels)
            for cls, count in sorted(dist.items()):
                print(f"    {cls:<35} +{count}")

        train_texts  = train_texts  + new_texts
        train_labels = train_labels + new_labels
        remaining[accepted_global] = False

        # ── Step 4: retrain ────────────────────────────────────────────────────
        X_train_new = transform(vec, train_texts)
        y_train_new = np.array(train_labels)

        model.set_params(warm_start=True, max_iter=100)
        model.fit(X_train_new, y_train_new)

        # ── Step 5: refit temperature on validation set ────────────────────────
        X_val_arr = transform(vec, X_texts_val)
        y_val_arr = np.array(y_val)
        T = fit_temperature(model, X_val_arr, y_val_arr, verbose=verbose)
        save_temperature(T)

        if verbose:
            print(f"  Retrained on {len(train_texts)} samples total\n")

        # ── Step 6: lower percentile bar ───────────────────────────────────────
        percentile = max(MIN_PERCENTILE, percentile - PERCENTILE_DECAY)

    # ── Persist ────────────────────────────────────────────────────────────────
    with open(SSL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"\nSSL model saved → {SSL_MODEL_PATH}")

    return model, T


if __name__ == "__main__":
    run(verbose=True)
