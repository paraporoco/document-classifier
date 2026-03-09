"""
metrics.py — Evaluation with calibrated confidence support.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.generator import CLASSES


def evaluate(model, X, y_true, split_name="Eval", T: Optional[float] = None):
    if T is not None:
        from features.calibration import calibrated_predict
        y_pred, confidence = calibrated_predict(model, X, T)
    else:
        y_pred     = model.predict(X)
        confidence = model.predict_proba(X).max(axis=1)

    print(f"\n{'─'*60}")
    print(f" {split_name}  ({len(y_true)} samples)")
    print(f"{'─'*60}")

    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=predicted):")
    _print_confusion(cm)

    print(f"\nConfidence distribution {'(calibrated)' if T else ''}:")
    bins   = [0.0, 0.5, 0.6, 0.7, 0.75, 0.85, 0.95, 1.01]
    labels = ["<0.50", "0.50–0.60", "0.60–0.70", "0.70–0.75", "0.75–0.85", "0.85–0.95", "≥0.95"]
    counts, _ = np.histogram(confidence, bins=bins)
    for lbl, count in zip(labels, counts):
        bar    = "█" * int(count / max(counts, default=1) * 30)
        pct    = count / len(confidence) * 100
        review = " ← review_flag" if lbl in ("<0.50","0.50–0.60","0.60–0.70","0.70–0.75") else ""
        print(f"  {lbl:<12}  {bar:<30}  {count:3d} ({pct:4.1f}%){review}")

    acc  = (np.array(y_pred) == np.array(y_true)).mean()
    print(f"\nOverall accuracy: {acc:.3f}")
    print(f"Mean confidence : {confidence.mean():.3f}")
    return {"accuracy": acc, "confidence_mean": confidence.mean()}


def compare(model_sup, model_ssl, X, y_true, T_sup=None, T_ssl=None):
    def acc(model, T):
        if T is not None:
            from features.calibration import calibrated_predict
            pred, _ = calibrated_predict(model, X, T)
        else:
            pred = model.predict(X)
        return (np.array(pred) == np.array(y_true)).mean()

    a_sup = acc(model_sup, T_sup)
    a_ssl = acc(model_ssl, T_ssl)
    delta = a_ssl - a_sup
    sign  = "+" if delta >= 0 else ""
    print(f"\n{'─'*40}")
    print(f"  Supervised only  : {a_sup:.3f}")
    print(f"  After SSL        : {a_ssl:.3f}  ({sign}{delta:.3f})")
    print(f"{'─'*40}")


def _print_confusion(cm):
    short = ["PUB","INT","CON","P_C","H_C","PHC"]
    print("       " + "  ".join(f"{s:>3}" for s in short))
    for i, row in enumerate(cm):
        cells = "  ".join(
            f"\033[1m{v:3d}\033[0m" if j == i else f"{v:3d}"
            for j, v in enumerate(row)
        )
        print(f"  {short[i]:>3}  {cells}")
