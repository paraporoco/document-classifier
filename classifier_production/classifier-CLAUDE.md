# CLAUDE.md — Real Training Data Integration

## Purpose of this session

Add real labelled documents to the document sensitivity classifier to improve accuracy
on genuine business content. The current model is trained entirely on synthetic data.
Real examples — even a small number — substantially improve generalisation.

---

## Project structure

```
classifier/
    main.py                        ← full pipeline: train + SSL + evaluate
    data/
        generator.py               ← synthetic data (CLASSES, SIGNALS, BOUNDARY_CASES, generate())
        real_examples.py           ← YOU WILL CREATE THIS — real labelled samples
    training/
        supervised.py              ← calls generate(), trains MLP, saves artefacts
    artefacts/
        model_ssl.pkl              ← trained model (replace after retraining)
        vectorizer.pkl             ← TF-IDF vectorizer (replace after retraining)
        temperature.pkl            ← temperature scalar (replace after retraining)

classifier_flat/
    app.py                         ← production CLI entrypoint
    predict.py                     ← inference (loads artefacts/)
    artefacts/                     ← copy new .pkl files here after retraining
```

---

## Classification hierarchy (policy v1.1)

| Code | Impact of unauthorised disclosure |
|---|---|
| `PUBLIC` | None. Authorised for publication. |
| `FOUO` | Noticeable. Internal use only, routine ops. |
| `CONFIDENTIAL` | Critical. Strategy, contracts, legal, financial projections. |
| `PERSONAL_CONFIDENTIAL` | CONFIDENTIAL + identifiable personal data. Floors here — cannot be FOUO. |
| `HIGHLY_CONFIDENTIAL` | Catastrophic. M&A, credentials, board materials, security findings. |
| `PERSONAL_HIGHLY_CONFIDENTIAL` | HIGHLY_CONFIDENTIAL + identifiable personal data. |

**Critical boundary rules:**
- A FOUO document containing any personal data (name, email, phone, DOB, ID number) → `PERSONAL_CONFIDENTIAL`
- The PERSONAL modifier does NOT escalate CONFIDENTIAL to HIGHLY_CONFIDENTIAL — that requires independent HC signals
- When in doubt between two adjacent levels, use the lower level

---

## Task: create `classifier/data/real_examples.py`

Create a new file `classifier/data/real_examples.py` with the following structure:

```python
"""
real_examples.py — Real labelled document samples.

Guidelines:
- Anonymise all personal data before adding (replace names, emails, IDs with placeholders)
- Keep text as close to the original as possible — only change identifying details
- Minimum 5 examples per class, aim for 10-20 per class
- Include examples at decision boundaries — these are most valuable:
    FOUO vs CONFIDENTIAL
    FOUO + PII → PERSONAL_CONFIDENTIAL
    CONFIDENTIAL vs HIGHLY_CONFIDENTIAL
    PERSONAL_CONFIDENTIAL vs PERSONAL_HIGHLY_CONFIDENTIAL
"""

from data.generator import Sample, CLASS_TO_IDX

REAL_EXAMPLES: list[Sample] = [
    Sample(
        text="...",
        label="FOUO",
        label_idx=CLASS_TO_IDX["FOUO"],
    ),
    # add more...
]
```

---

## Task: integrate real examples into the training pipeline

Once `real_examples.py` exists, edit `classifier/training/supervised.py` to inject real examples:

Find this block (around line 48):
```python
labelled, _ = generate(n_labelled=n_labelled, seed=seed)
```

Replace with:
```python
labelled, _ = generate(n_labelled=n_labelled, seed=seed)

# Inject real examples — appended after synthetic, before shuffle
try:
    from data.real_examples import REAL_EXAMPLES
    labelled.extend(REAL_EXAMPLES)
    if verbose:
        from collections import Counter
        counts = Counter(s.label for s in REAL_EXAMPLES)
        print(f"  Real examples injected: {len(REAL_EXAMPLES)}")
        for cls, n in sorted(counts.items()):
            print(f"    {cls}: {n}")
except ImportError:
    pass  # real_examples.py not present — silently skip
```

Do the same in `classifier/training/supervised.py` inside `train_with_calibration()` if it
has a separate call to `generate()`.

Also check `classifier/training/pseudo_label.py` — the SSL unlabelled pool is generated
separately. Real examples should NOT be added to the unlabelled pool (they are labelled).

---

## Task: retrain after adding examples

```bash
cd classifier
python main.py
```

Then copy artefacts to the production folder:

```bash
cp classifier/artefacts/model_ssl.pkl   classifier_flat/artefacts/
cp classifier/artefacts/vectorizer.pkl  classifier_flat/artefacts/
cp classifier/artefacts/temperature.pkl classifier_flat/artefacts/
```

---

## Quality checks after retraining

Run this quick check to verify the model handles boundary cases correctly:

```python
# Run from classifier_flat/
from predict import predict, load
load()

tests = [
    # FOUO+PII must floor at PERSONAL_CONFIDENTIAL
    ("FOR OFFICIAL USE ONLY. Staff list. Jane Smith, j.smith@org.com, +44 7700 900123. Internal use only.", "PERSONAL_CONFIDENTIAL"),
    # FOUO stays FOUO without PII
    ("FOR OFFICIAL USE ONLY. Weekly team sync notes. No blockers. Next review Friday. Internal use only.", "FOUO"),
    # HC stays HC
    ("HIGHLY CONFIDENTIAL. Proposed acquisition. Valuation 200M EUR. Due diligence in progress. Board approval required.", "HIGHLY_CONFIDENTIAL"),
]

for text, expected in tests:
    r = predict(text)
    ok = "v" if r["classification"] == expected else "X"
    print(f"[{ok}] {expected} -> {r['classification']} (conf={r['confidence']:.3f})")
```

---

## Anonymisation rules for real documents

Before adding any real document text:

| Data type | Replace with |
|---|---|
| Person names | `[Name]` or generic placeholder e.g. `John Smith` |
| Email addresses | `name@organisation.com` |
| Phone numbers | `+XX XXXX XXXXXX` |
| Employee / customer IDs | `EMP-XXXXX` / `CUS-XXXXX` |
| Company names (counterparty) | `[Company]` or `[REDACTED]` |
| Financial figures | Keep order of magnitude, change exact number (e.g. €342M → €300M) |
| Dates of birth | Change day/month, keep year |
| Addresses | `[Address]` |

The label must reflect the **original** document's sensitivity — not the anonymised version.

---

## What makes a good real example

**High value:**
- Documents at a boundary that was previously misclassified
- Documents with mixed signals (e.g. FOUO label + PII content = PERSONAL_CONFIDENTIAL)
- Short fragments (email subjects, document headers, footer stamps)
- Documents in languages other than English

**Lower value:**
- Documents with a single obvious explicit label and no other content
- Documents that are already well-represented by synthetic examples

---

## Do not modify

- `classifier/data/generator.py` — synthetic data is separate from real data
- `classifier_flat/predict.py` — inference layer is not affected by training data changes
- `classifier_flat/app.py` — production CLI is not affected
- `classifier.spec` — PyInstaller build config is not affected

---

## After adding real examples — rebuild the exe

```bat
cd classifier_flat
pyinstaller classifier.spec
```

Then test:

```bat
dist\classify.exe path\to\test_document.pdf
```

---

## Commit to GitHub

```bash
git add classifier/data/real_examples.py classifier/artefacts/ classifier_flat/artefacts/
git commit -m "Add real training examples — N examples across X classes"
git push origin main
```

Do NOT commit `dist/` or `build/` — they are in `.gitignore`.
