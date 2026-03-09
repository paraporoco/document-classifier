"""
main.py — Full pipeline orchestrator.

Phases:
    1. Supervised training + temperature calibration
    2. Evaluation (calibrated confidence)
    3. Semi-supervised pseudo-labelling (percentile-based threshold)
    4. Before/after comparison
    5. Inference demo
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from data.generator import CLASSES, generate
from features.vectorizer import load as load_vec, transform
from features.calibration import load_temperature, calibrated_predict
from training.supervised import train_with_calibration
from training.pseudo_label import run as ssl_run
from evaluation.metrics import evaluate, compare
from inference.predict import predict_one

ARTEFACTS = Path(__file__).parent / "artefacts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ssl", action="store_true")
    parser.add_argument("--infer", type=str, default=None)
    args = parser.parse_args()

    if args.infer:
        predict_one(args.infer, ssl=False, verbose=True)
        return

    print("=" * 60)
    print(" DOCUMENT CLASSIFICATION — FULL PIPELINE")
    print("=" * 60)

    # ── Phase 1: Supervised training + calibration ─────────────────────────────
    print("\n[1/4] Supervised training + temperature calibration")
    model_sup, vec, X_val, y_val, T = train_with_calibration(verbose=True)

    # ── Phase 2: Calibrated evaluation ────────────────────────────────────────
    print("\n[2/4] Supervised model evaluation (calibrated)")
    evaluate(model_sup, X_val, y_val, split_name="Supervised (val)", T=T)

    if args.skip_ssl:
        _inference_demo(ssl=False)
        return

    # ── Phase 3: Semi-supervised pseudo-labelling ──────────────────────────────
    print("\n[3/4] Semi-supervised self-training (percentile threshold)")
    model_ssl, T_ssl = ssl_run(verbose=True)

    # ── Phase 4: Before / after comparison ────────────────────────────────────
    print("\n[4/4] Comparison: supervised vs. semi-supervised")
    labelled, _ = generate()
    texts  = [s.text      for s in labelled]
    labels = [s.label_idx for s in labelled]
    _, X_texts_val, _, y_val_arr = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val_cmp = transform(vec, X_texts_val)
    y_val_cmp = np.array(y_val_arr)

    evaluate(model_ssl, X_val_cmp, y_val_cmp, split_name="After SSL (val)", T=T_ssl)
    compare(model_sup, model_ssl, X_val_cmp, y_val_cmp, T_sup=T, T_ssl=T_ssl)

    # ── Phase 5: Inference demo ────────────────────────────────────────────────
    _inference_demo(ssl=True)


def _inference_demo(ssl: bool = True) -> None:
    DEMO = [
        ("For immediate release. No restriction. Open to all stakeholders.", "PUBLIC"),
        ("Internal use only. Do not distribute externally. Headcount planning figures.", "FOUO"),
        ("Non-disclosure agreement. Business strategy. Financial projection. Under embargo.", "CONFIDENTIAL"),
        ("Employee record. Date of birth. Home address. Customer personal data.", "PERSONAL_CONFIDENTIAL"),
        ("The proposed acquisition of [REDACTED] at a valuation of €340M requires board approval.", "HIGHLY_CONFIDENTIAL"),
        ("Strictly confidential. Employee ID EMP-00421. Named individual performance review. Compensation adjustment +4.2%.", "PERSONAL_HIGHLY_CONFIDENTIAL"),
    ]
    label = "SSL" if ssl else "Supervised"
    print(f"\n{'='*60}\n INFERENCE DEMO  ({label} model)\n{'='*60}")
    correct = 0
    for text, expected in DEMO:
        r = predict_one(text, ssl=ssl)
        match = r["classification"] == expected
        correct += int(match)
        flag = "✓" if match else f"✗  expected: {expected}"
        print(f"\n  {flag}")
        print(f"  Input     : {text[:75]}")
        print(f"  Predicted : {r['classification']}  (conf={r['confidence']:.3f}{'  ⚑' if r['review_flag'] else ''})")
        print(f"  Reasoning : {r['reasoning'][:100]}")
    print(f"\n  Demo accuracy: {correct}/{len(DEMO)}")
if __name__ == "__main__":
    main()
