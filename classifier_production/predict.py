import pickle, sys
from pathlib import Path
import numpy as np

CLASSES = ["PUBLIC","FOUO","CONFIDENTIAL","PERSONAL_CONFIDENTIAL","HIGHLY_CONFIDENTIAL","PERSONAL_HIGHLY_CONFIDENTIAL"]
REVIEW_THRESHOLD = 0.80
POLICY_VERSION   = "1.1"

def _artefact(name):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / "artefacts" / name

def _load():
    with open(_artefact("model_ssl.pkl"),  "rb") as f: model = pickle.load(f)
    with open(_artefact("vectorizer.pkl"), "rb") as f: vec   = pickle.load(f)
    with open(_artefact("temperature.pkl"),"rb") as f: T     = pickle.load(f)
    return model, vec, T

_model = _vec = _T = None

def load():
    global _model, _vec, _T
    _model, _vec, _T = _load()

def _calibrated(model, X, T):
    proba = model.predict_proba(X)
    logits = np.log(proba + 1e-12)
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    proba_cal = exp / exp.sum(axis=1, keepdims=True)
    return proba_cal.argmax(axis=1), proba_cal.max(axis=1)

def _reasoning(x):
    names   = np.array(_vec.get_feature_names_out())
    weights = np.abs(_model.coefs_[0]).sum(axis=1)
    contrib = x * weights
    top     = contrib.argsort()[-5:][::-1]
    feats   = [f"{names[i]} ({contrib[i]:.3f})" for i in top if contrib[i] > 0]
    return ("Top signal features: " + ", ".join(feats)) if feats else "No strong signals."

def predict(text: str) -> dict:
    X = _vec.transform([text]).toarray().astype(np.float32)
    pred, conf = _calibrated(_model, X, _T)
    idx = int(pred[0]); confidence = float(conf[0])
    return {
        "classification" : CLASSES[idx],
        "confidence"     : round(confidence, 4),
        "review_flag"    : confidence < REVIEW_THRESHOLD,
        "reasoning"      : _reasoning(X[0]),
        "policy_version" : POLICY_VERSION,
    }
