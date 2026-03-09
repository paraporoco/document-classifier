"""
calibration.py — Post-hoc confidence calibration via temperature scaling.

Problem:
    A neural network's softmax outputs are not calibrated probabilities.
    At high noise rates, the model hedges: it's still accurate but its
    max-softmax confidence is spread flat (mean ~0.3 at 50% noise).
    The review_flag threshold becomes useless — everything gets flagged.

Solution — Temperature Scaling:
    Divide raw logits by a scalar T before applying softmax.
        p = softmax(logits / T)

    T > 1  →  flattens distribution  →  lowers overconfident scores
    T < 1  →  sharpens distribution  →  raises underconfident scores

    T is fit by minimising Negative Log-Likelihood on the validation set.
    One parameter. No retraining. Works post-hoc on any sklearn MLP.

Limitation:
    sklearn's MLPClassifier does not expose raw logits directly.
    We recover approximate logits by applying the inverse of the final
    softmax: logits ≈ log(p + ε). This is an approximation but sufficient
    for calibration purposes on this problem scale.

Reference:
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    https://arxiv.org/abs/1706.04599
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import CLASSES, generate
from features.vectorizer import load as load_vec, transform

ARTEFACTS   = Path(__file__).parent.parent / "artefacts"
CALIB_PATH  = ARTEFACTS / "temperature.pkl"


def _recover_logits(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Approximate logits from softmax probabilities.
    logits ≈ log(p)  (up to a constant shift, which cancels in softmax).
    """
    return np.log(proba + eps)


def _calibrated_proba(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to logits and return softmax probabilities."""
    scaled = logits / T
    # Numerically stable softmax
    scaled -= scaled.max(axis=1, keepdims=True)
    exp    = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(T: float, logits: np.ndarray, y_true: np.ndarray) -> float:
    """Negative log-likelihood of calibrated predictions."""
    proba = _calibrated_proba(logits, T)
    # Clip to avoid log(0)
    correct_proba = proba[np.arange(len(y_true)), y_true].clip(1e-12, 1.0)
    return -np.log(correct_proba).mean()


def fit_temperature(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> float:
    """
    Find the optimal temperature T that minimises NLL on the validation set.

    Returns T (float). T > 1 means the model was overconfident.
                       T < 1 means the model was underconfident.
    """
    proba_raw = model.predict_proba(X_val)
    logits    = _recover_logits(proba_raw)

    result = minimize_scalar(
        _nll,
        args=(logits, y_val),
        bounds=(0.01, 10.0),
        method="bounded",
    )
    T = result.x

    # Diagnostics
    conf_before = proba_raw.max(axis=1).mean()
    proba_cal   = _calibrated_proba(logits, T)
    conf_after  = proba_cal.max(axis=1).mean()

    if verbose:
        print(f"\nTemperature scaling:")
        print(f"  T = {T:.4f}  ({'sharpening' if T < 1 else 'flattening'})")
        print(f"  Mean confidence before : {conf_before:.3f}")
        print(f"  Mean confidence after  : {conf_after:.3f}")

    return T


def save_temperature(T: float, path: Path = CALIB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(T, f)
    print(f"  Temperature saved → {path}  (T={T:.4f})")


def load_temperature(path: Path = CALIB_PATH) -> float:
    with open(path, "rb") as f:
        return pickle.load(f)


def calibrated_predict(
    model,
    X: np.ndarray,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (predicted_class_idx, calibrated_confidence) for each row in X.
    """
    proba_raw  = model.predict_proba(X)
    logits     = _recover_logits(proba_raw)
    proba_cal  = _calibrated_proba(logits, T)
    predicted  = proba_cal.argmax(axis=1)
    confidence = proba_cal.max(axis=1)
    return predicted, confidence


if __name__ == "__main__":
    from training.supervised import load_model

    model = load_model()
    vec   = load_vec()

    labelled, _ = generate()
    texts  = [s.text      for s in labelled]
    labels = [s.label_idx for s in labelled]

    _, X_texts_val, _, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val = transform(vec, X_texts_val)
    y_val = np.array(y_val)

    T = fit_temperature(model, X_val, y_val)
    save_temperature(T)
