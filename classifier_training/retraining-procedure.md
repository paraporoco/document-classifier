# Model Retraining Procedure

**Project:** document-classifier  
**Applies to:** `classifier_training/`  
**Policy version:** 1.1

---

## Overview

Retraining improves the model by adding real labelled documents to supplement the synthetic training data. The output is three artefact files deployed to two targets: the desktop executable and the web server.

```
classifier_training/        classifier_production/              classifier_web/
    main.py          →           artefacts/                         web_artefacts/
    artefacts/*.pkl  →               model_ssl.pkl    →  convert  →     model.<hash>.onnx
                                     vectorizer.pkl   →  artefacts →    vectorizer.<hash>.json
                                     temperature.pkl  →             →    temperature.<hash>.json
                                                                         manifest.json
                              pyinstaller classifier.spec
                                     ↓
                              classify.exe (desktop)              uvicorn server:app (web)
```

---

## Step 1 — Extract text from your documents

Use the extraction layer already present in `classifier_production/` to get clean text from any file:

```python
# extract_for_training.py — run once to dump extracted text
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pdf_extractor import read as read_pdf
from office_extractor import read as read_office
from text_extractor import read as read_text

def extract(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        text, status, _ = read_pdf(p)
    elif ext in (".docx", ".xlsx", ".pptx"):
        text, status, _ = read_office(p)
    elif ext in (".txt", ".md", ".html", ".htm"):
        text, status, _ = read_text(p)
    else:
        return None
    return text if status == "ok" else None

print(extract("path/to/your/document.pdf"))
```

---

## Step 2 — Label each document

### Documents with a marking footer

The footer is the starting point, not the final answer. Apply this logic:

```
Has footer label?
├── YES → Start from the footer level
│         Does the content contain personal data?
│         (name, ID, DOB, email, phone, compensation, health data)
│         ├── YES + footer is FOUO          → PERSONAL_CONFIDENTIAL
│         ├── YES + footer is CONFIDENTIAL  → PERSONAL_CONFIDENTIAL
│         ├── YES + footer is HC            → PERSONAL_HIGHLY_CONFIDENTIAL
│         └── NO                            → keep footer level
└── NO  → Classify by content signals (see table below)
```

### Documents without a marking footer

| Content signals | Label |
|---|---|
| "press release", "for immediate release", "no restriction" | `PUBLIC` |
| "internal use only", "FOUO", "not for external distribution", routine ops content | `FOUO` |
| "confidential", NDA, contracts, strategy docs, financial projections, legal matters | `CONFIDENTIAL` |
| Any CONFIDENTIAL content + names / emails / phone / DOB / IDs | `PERSONAL_CONFIDENTIAL` |
| "highly confidential", M&A, security credentials, board materials, source code | `HIGHLY_CONFIDENTIAL` |
| Any HC content + personal data about a named individual | `PERSONAL_HIGHLY_CONFIDENTIAL` |

When genuinely unsure between two adjacent levels, assign the lower one.

---

## Step 3 — Add documents to `real_examples.py`

Edit `classifier_training/data/real_examples.py`. Each document is one `Sample` entry:

```python
from data.generator import Sample, CLASS_TO_IDX

REAL_EXAMPLES: list[Sample] = [

    # Document with CONFIDENTIAL footer, no PII
    Sample(
        text="""CONFIDENTIAL

        Q3 Strategic Review — draft for board discussion.
        Revenue outlook revised upward to €85M. Market entry into DACH region
        accelerated to Q1 2025. Partnership discussions with [REDACTED] ongoing.
        Do not circulate outside the executive committee.""",
        label="CONFIDENTIAL",
        label_idx=CLASS_TO_IDX["CONFIDENTIAL"],
    ),

    # FOUO footer but contains PII — upgraded to PERSONAL_CONFIDENTIAL
    Sample(
        text="""FOR OFFICIAL USE ONLY

        Staff directory update — London office.
        Ana Costa | a.costa@org.com | +44 7700 900234 | joined 12 Jan 2023
        Peter Bauer | p.bauer@org.com | +44 7700 900567 | joined 3 Mar 2022
        Not for external distribution.""",
        label="PERSONAL_CONFIDENTIAL",
        label_idx=CLASS_TO_IDX["PERSONAL_CONFIDENTIAL"],
    ),

    # No footer — classified by content
    Sample(
        text="""Weekly sync — product team
        Sprint 14 demo scheduled for Thursday.
        No blockers reported. Backlog grooming moved to next week.""",
        label="FOUO",
        label_idx=CLASS_TO_IDX["FOUO"],
    ),

]
```

**Anonymise before adding.** Replace identifying details with placeholders — the label reflects the original sensitivity, not the anonymised text:

| Original | Placeholder |
|---|---|
| Real names | `[Name]` |
| Email addresses | `name@organisation.com` |
| Phone numbers | `+XX XXX XXX XXXX` |
| Employee IDs | `EMP-XXXXX` |
| Organisation names | `[Organisation]` |

---

## Step 4 — Retrain

```powershell
cd D:\Projects\document_classifier\classifier_training
python main.py
```

Expected output:

```
============================================================
 DOCUMENT CLASSIFICATION — FULL PIPELINE
============================================================

[1/4] Supervised training + temperature calibration
  Real examples injected: <N>
    CONFIDENTIAL: <n>
    FOUO: <n>
    ...
  Train : <N> samples | <N> features
  Val   : <N> samples

[2/4] Supervised model evaluation (calibrated)
[3/4] Semi-supervised self-training (percentile threshold)
[4/4] Comparison: supervised vs. semi-supervised
```

---

## Step 5 — Copy artefacts to production

```powershell
cd D:\Projects\document_classifier

copy classifier_training\artefacts\model_ssl.pkl   classifier_production\artefacts\
copy classifier_training\artefacts\vectorizer.pkl  classifier_production\artefacts\
copy classifier_training\artefacts\temperature.pkl classifier_production\artefacts\
```

---

## Step 6 — Rebuild the exe

```powershell
cd classifier_production
pyinstaller classifier.spec
```

The new `classify.exe` is in `classifier_production\dist\`.

---

## Step 7 — Regenerate web artefacts

Converts the new pkl files to browser-compatible ONNX + JSON for the web server.
Content hashes in the filenames allow the browser to cache artefacts permanently;
`manifest.json` tells the Service Worker which files are current.

```powershell
cd classifier_production
python convert_artefacts.py
```

The script prints a built-in ONNX smoke-test confirming the output shape is correct.

**Additional parity check** — confirm web inference matches the Python results:

```python
# Run from classifier_production/
import onnxruntime as rt, json, numpy as np, pickle, pathlib

artefacts = pathlib.Path("../classifier_web/web_artefacts")
manifest  = json.loads((artefacts / "manifest.json").read_text())

sess = rt.InferenceSession(str(artefacts / manifest["model"]))
with open("artefacts/vectorizer.pkl", "rb") as f: vec = pickle.load(f)

tests = [
    ("HIGHLY CONFIDENTIAL. Proposed acquisition. Valuation 200M EUR. Board approval required.", "HIGHLY_CONFIDENTIAL"),
    ("FOR OFFICIAL USE ONLY. Staff list. Jane Smith, j.smith@org.com. Internal use only.",       "PERSONAL_CONFIDENTIAL"),
    ("Weekly team sync notes. No blockers. Next review Friday. Internal use only.",               "FOUO"),
]
classes = manifest["classes"]
for text, expected in tests:
    X   = vec.transform([text]).toarray().astype("float32")
    out = sess.run(None, {"float_input": X})
    idx = out[1][0].argmax()
    ok  = "v" if classes[idx] == expected else "X"
    print(f"[{ok}] expected={expected}  got={classes[idx]}")
```

---

## Step 8 — Commit to repo

`classifier_web/web_artefacts/` is gitignored — it is generated at deployment time,
not stored in source control. Commit only training data and pkl artefacts:

```powershell
cd D:\Projects\document_classifier

git add classifier_training/data/real_examples.py
git add classifier_training/artefacts/
git add classifier_production/artefacts/

git commit -m "retrain: add <N> real examples across <X> classes"
git push origin main
```

---

## How many documents you need

| Count per class | Effect |
|---|---|
| 1–5 | Marginal improvement on that class |
| 5–15 | Noticeable improvement at classification boundaries |
| 15–30 | Strong improvement; model generalises to real phrasing |
| 30+ | Diminishing returns unless the model architecture is also upgraded |

**Boundary examples are worth 3–5× more than obvious easy ones.** Priority boundaries:

- FOUO document containing PII → should become `PERSONAL_CONFIDENTIAL`
- FOUO vs CONFIDENTIAL (impact threshold ambiguity)
- CONFIDENTIAL vs HIGHLY_CONFIDENTIAL
- PERSONAL_CONFIDENTIAL vs PERSONAL_HIGHLY_CONFIDENTIAL

---

## Handling unlabelled documents

For documents where you cannot confidently determine the label, add them to the unlabelled pool instead:

```python
# In real_examples.py
REAL_UNLABELLED: list[Sample] = [
    Sample(
        text="...",
        label=None,
        label_idx=None,
    ),
]
```

Then in `classifier_training/training/supervised.py`, extend the unlabelled pool the same way as the labelled pool. The SSL phase will pseudo-label these automatically.

A wrong label is worse than no label. When in doubt, use the unlabelled pool.

---

## Full directory reference

```
classifier_training/
    main.py                      entry point — run to retrain
    requirements.txt
    data/
        generator.py             synthetic data generator (do not edit for retraining)
        real_examples.py         ← edit this with your documents
    features/
        vectorizer.py
        calibration.py
    training/
        supervised.py            injects real_examples automatically
        pseudo_label.py
    model/
        net.py
    evaluation/
        metrics.py
    inference/
        predict.py
    artefacts/
        model_ssl.pkl            ← output: copy to classifier_production/artefacts/
        vectorizer.pkl           ← output: copy to classifier_production/artefacts/
        temperature.pkl          ← output: copy to classifier_production/artefacts/

classifier_production/
    artefacts/                   ← copy pkl files here after retraining
    convert_artefacts.py         ← run after copying pkl files (Step 7)
    app.py                       CLI entrypoint (do not modify)
    predict.py                   inference (do not modify)
    classifier.spec              PyInstaller config (do not modify)
    dist/classify.exe            ← rebuilt by pyinstaller classifier.spec

classifier_web/
    server.py                    FastAPI server
    web/                         browser SPA (static files)
    web_artefacts/               ← generated by convert_artefacts.py (gitignored)
        manifest.json
        model.<hash>.onnx
        vectorizer.<hash>.json
        temperature.<hash>.json
```
