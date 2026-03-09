"""
supervised.py — Supervised training on the labelled pool.

Pipeline:
    1. Generate labelled samples
    2. Fit TF-IDF vectorizer on training split only
    3. Train MLP, print per-epoch loss curve
    4. Evaluate on held-out validation split
    5. Persist model + vectorizer

PyTorch equivalent training loop is annotated inline.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import CLASSES, generate
from features.vectorizer import fit as fit_vec, save as save_vec, transform
from model.net import build_network
from evaluation.metrics import evaluate

ARTEFACTS = Path(__file__).parent.parent / "artefacts"
MODEL_PATH = ARTEFACTS / "model.pkl"


def train(
    n_labelled: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Full supervised training run.

    Returns:
        model     — fitted MLPClassifier
        vec       — fitted TfidfVectorizer
        X_val     — validation features
        y_val     — validation labels (int)
    """
    # ── 1. Data ────────────────────────────────────────────────────────────────
    labelled, _ = generate(n_labelled=n_labelled, seed=seed)

    try:
        from data.real_examples import REAL_EXAMPLES
        labelled.extend(REAL_EXAMPLES)
        if verbose and REAL_EXAMPLES:
            from collections import Counter
            counts = Counter(s.label for s in REAL_EXAMPLES)
            print(f"  Real examples injected: {len(REAL_EXAMPLES)}")
            for cls, n in sorted(counts.items()):
                print(f"    {cls}: {n}")
    except ImportError:
        pass

    texts  = [s.text      for s in labelled]
    labels = [s.label_idx for s in labelled]

    X_texts_train, X_texts_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.2,
        stratify=labels,    # preserve class balance in both splits
        random_state=seed,
    )

    # ── 2. Features ────────────────────────────────────────────────────────────
    # Fit ONLY on training data — vectorizer must not see validation text.
    #
    # PyTorch equivalent:
    #   (same: fit sklearn vectorizer on train, transform both splits)
    vec     = fit_vec(X_texts_train)
    X_train = transform(vec, X_texts_train)
    X_val   = transform(vec, X_texts_val)
    y_train = np.array(y_train)
    y_val   = np.array(y_val)

    if verbose:
        print(f"Train : {X_train.shape[0]} samples | {X_train.shape[1]} features")
        print(f"Val   : {X_val.shape[0]} samples")

    # ── 3. Model ───────────────────────────────────────────────────────────────
    #
    # PyTorch equivalent:
    #   model = ClassifierNet(input_dim=X_train.shape[1])
    #   criterion = nn.CrossEntropyLoss()
    #   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #   for epoch in range(max_epochs):
    #       model.train()
    #       for xb, yb in DataLoader(train_dataset, batch_size=32, shuffle=True):
    #           optimizer.zero_grad()
    #           loss = criterion(model(xb), yb)
    #           loss.backward()
    #           optimizer.step()
    #       # early stopping check on val loss
    model = build_network()

    if verbose:
        print("\nTraining …")

    model.fit(X_train, y_train)

    # ── 4. Report ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"Epochs run       : {model.n_iter_}")
        print(f"Final train loss : {model.loss_:.4f}")
        evaluate(model, X_val, y_val, split_name="Validation")

    # ── 5. Persist ─────────────────────────────────────────────────────────────
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    save_vec(vec)

    if verbose:
        print(f"\nModel saved → {MODEL_PATH}")

    return model, vec, X_val, y_val


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train(verbose=True)


def train_with_calibration(n_labelled: int = 100, seed: int = 42, verbose: bool = True):
    """train() + temperature scaling fit in one call."""
    from features.calibration import fit_temperature, save_temperature
    model, vec, X_val, y_val = train(n_labelled=n_labelled, seed=seed, verbose=verbose)
    T = fit_temperature(model, X_val, y_val, verbose=verbose)
    save_temperature(T)
    return model, vec, X_val, y_val, T
