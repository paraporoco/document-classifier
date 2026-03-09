"""
convert_artefacts.py — Convert sklearn training artefacts to browser-compatible formats.

Run this after retraining and before deploying the web server:

    cd classifier_production
    python convert_artefacts.py

Outputs to ../classifier_web/web_artefacts/ (configurable via --out-dir):
    model.<hash>.onnx          MLP weights in ONNX format (for onnxruntime-web)
    vectorizer.<hash>.json     TF-IDF vocabulary, IDF weights, coef weights
    temperature.<hash>.json    Temperature calibration scalar
    manifest.json              Index of current artefact filenames + metadata

Content hashes in filenames allow long-lived browser caching (immutable).
The manifest is always fetched fresh; it points to the current hashed files.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

CLASSES = [
    "PUBLIC", "FOUO", "CONFIDENTIAL",
    "PERSONAL_CONFIDENTIAL", "HIGHLY_CONFIDENTIAL", "PERSONAL_HIGHLY_CONFIDENTIAL",
]
POLICY_VERSION = "1.1"
REVIEW_THRESHOLD = 0.80


def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def load_artefacts(artefacts_dir: Path):
    with open(artefacts_dir / "model_ssl.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artefacts_dir / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(artefacts_dir / "temperature.pkl", "rb") as f:
        T = pickle.load(f)
    return model, vec, T


def convert_model(model, n_features: int, out_dir: Path) -> str:
    """Export MLPClassifier to ONNX with zipmap=False for flat float32 probability output."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    options = {id(model): {"zipmap": False}}
    onnx_model = convert_sklearn(
        model, initial_types=initial_type, target_opset=17, options=options,
    )
    data = onnx_model.SerializeToString()
    h = _sha8(data)
    path = out_dir / f"model.{h}.onnx"
    path.write_bytes(data)
    print(f"  model       → {path.name}  ({len(data)/1024:.1f} KB)")
    return path.name


def convert_vectorizer(model, vec, out_dir: Path) -> str:
    """
    Export TF-IDF vectorizer to JSON.

    Includes:
      vocabulary    — dict: token -> feature index
      idf           — list of IDF weights (one per feature, in feature-index order)
      coef_weights  — |coefs_[0]|.sum(axis=1): used for top-feature reasoning in JS
    """
    coef_weights = np.abs(model.coefs_[0]).sum(axis=1).tolist()
    data = {
        "vocabulary":    {k: int(v) for k, v in vec.vocabulary_.items()},  # numpy int64 → Python int
        "idf":           vec.idf_.tolist(),          # float per feature
        "coef_weights":  coef_weights,               # float per feature (for reasoning)
        "ngram_range":   list(vec.ngram_range),
        "sublinear_tf":  vec.sublinear_tf,
        "norm":          vec.norm,
    }
    raw = json.dumps(data, separators=(",", ":")).encode()
    h = _sha8(raw)
    path = out_dir / f"vectorizer.{h}.json"
    path.write_bytes(raw)
    print(f"  vectorizer  → {path.name}  ({len(raw)/1024:.1f} KB, {len(vec.vocabulary_)} tokens)")
    return path.name


def convert_temperature(T, out_dir: Path) -> str:
    raw = json.dumps({"T": float(T)}, separators=(",", ":")).encode()
    h = _sha8(raw)
    path = out_dir / f"temperature.{h}.json"
    path.write_bytes(raw)
    print(f"  temperature → {path.name}  (T={float(T):.4f})")
    return path.name


def write_manifest(out_dir: Path, model_file: str, vec_file: str, temp_file: str) -> None:
    manifest = {
        "model":            model_file,
        "vectorizer":       vec_file,
        "temperature":      temp_file,
        "classes":          CLASSES,
        "policy_version":   POLICY_VERSION,
        "review_threshold": REVIEW_THRESHOLD,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest    → {path.name}")


def validate_onnx(out_dir: Path, model_file: str, vec) -> None:
    """Quick smoke-test: run ONNX inference with a synthetic vector and confirm output shape."""
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(str(out_dir / model_file))
        dummy = vec.transform(["test document"]).toarray().astype("float32")
        out = sess.run(None, {"float_input": dummy})
        # out[0] = labels, out[1] = probabilities [1, n_classes]
        assert out[1].shape == (1, len(CLASSES)), f"Unexpected output shape: {out[1].shape}"
        print(f"  ONNX check  → OK (output shape {out[1].shape})")
    except ImportError:
        print("  ONNX check  → skipped (onnxruntime not installed; install for validation)")
    except Exception as e:
        print(f"  ONNX check  → FAILED: {e}")
        raise


def main():
    ap = argparse.ArgumentParser(
        description="Convert sklearn artefacts to browser-compatible ONNX + JSON.",
    )
    ap.add_argument(
        "--artefacts-dir", type=Path,
        default=Path(__file__).parent / "artefacts",
        help="Directory containing model_ssl.pkl, vectorizer.pkl, temperature.pkl",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent.parent / "classifier_web" / "web_artefacts",
        help="Output directory for browser-compatible artefacts",
    )
    ap.add_argument(
        "--no-validate", action="store_true",
        help="Skip ONNX smoke-test (requires onnxruntime)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading artefacts…")
    model, vec, T = load_artefacts(args.artefacts_dir)

    n_features = model.coefs_[0].shape[0]
    print(f"  Model input : {n_features} features, {len(CLASSES)} classes")
    print(f"  Architecture: {[l.shape for l in model.coefs_]}")

    print("\nConverting…")
    model_file = convert_model(model, n_features, args.out_dir)
    vec_file   = convert_vectorizer(model, vec, args.out_dir)
    temp_file  = convert_temperature(T, args.out_dir)
    write_manifest(args.out_dir, model_file, vec_file, temp_file)

    if not args.no_validate:
        print("\nValidating ONNX…")
        validate_onnx(args.out_dir, model_file, vec)

    print(f"\nDone. Artefacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
