"""
predict.py — Single-document and batch inference with calibrated confidence.

Output schema matches classifier prompt v0.3:
    {
        "classification"  : str,
        "confidence"      : float,   # calibrated (temperature-scaled) probability
        "review_flag"     : bool,    # True when confidence < REVIEW_THRESHOLD
        "reasoning"       : str,
        "policy_version"  : str
    }
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import CLASSES
from features.vectorizer import load as load_vec, transform
from features.calibration import calibrated_predict, load_temperature, _recover_logits, _calibrated_proba

ARTEFACTS        = Path(__file__).parent.parent / "artefacts"
REVIEW_THRESHOLD = 0.75
POLICY_VERSION   = "1.0"
N_REASON_FEATURES = 5


def _load_model(ssl: bool = False):
    import pickle
    path = ARTEFACTS / ("model_ssl.pkl" if ssl else "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _reasoning(vec, model, x_dense: np.ndarray, T: float, predicted_class: int) -> str:
    feature_names      = np.array(vec.get_feature_names_out())
    first_layer_weights = np.abs(model.coefs_[0]).sum(axis=1)
    contribution        = x_dense * first_layer_weights
    top_idx = contribution.argsort()[-N_REASON_FEATURES:][::-1]
    top_features = [
        f"{feature_names[i]} ({contribution[i]:.3f})"
        for i in top_idx
        if contribution[i] > 0
    ]
    if not top_features:
        return "No strong signal features detected."
    return "Top signal features: " + ", ".join(top_features)


def predict_one(text: str, ssl: bool = False, verbose: bool = False) -> dict:
    vec   = load_vec()
    model = _load_model(ssl=ssl)
    T     = load_temperature()

    x          = transform(vec, [text])
    pred, conf = calibrated_predict(model, x, T)
    pred_idx   = int(pred[0])
    confidence = float(conf[0])

    result = {
        "classification" : CLASSES[pred_idx],
        "confidence"     : round(confidence, 4),
        "review_flag"    : confidence < REVIEW_THRESHOLD,
        "reasoning"      : _reasoning(vec, model, x[0], T, pred_idx),
        "policy_version" : POLICY_VERSION,
    }
    if verbose:
        _print_result(result)
    return result


def predict_batch(texts: list[str], ssl: bool = False) -> list[dict]:
    vec   = load_vec()
    model = _load_model(ssl=ssl)
    T     = load_temperature()

    X          = transform(vec, texts)
    preds, confs = calibrated_predict(model, X, T)

    return [
        {
            "classification" : CLASSES[int(preds[i])],
            "confidence"     : round(float(confs[i]), 4),
            "review_flag"    : float(confs[i]) < REVIEW_THRESHOLD,
            "reasoning"      : _reasoning(vec, model, X[i], T, int(preds[i])),
            "policy_version" : POLICY_VERSION,
        }
        for i in range(len(texts))
    ]


def _print_result(r: dict) -> None:
    flag = "⚑ REVIEW" if r["review_flag"] else "✓"
    print(f"\n  classification : {r['classification']}")
    print(f"  confidence     : {r['confidence']:.4f}  {flag}")
    print(f"  review_flag    : {r['review_flag']}")
    print(f"  reasoning      : {r['reasoning']}")
    print(f"  policy_version : {r['policy_version']}")
