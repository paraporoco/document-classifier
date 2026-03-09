"""
real_examples.py — Add your own labelled documents here.

Each Sample has:
    text       str   — extracted document text (anonymised)
    label      str   — one of the six class names
    label_idx  int   — index matching CLASS_TO_IDX

Anonymise before adding:
    names       → [Name]
    emails      → name@organisation.com
    phone nos.  → +XX XXX XXX XXXX
    IDs         → EMP-XXXXX
    org names   → [Organisation]

Footer vs content conflicts:
    If the footer says FOUO but the document contains PII, use PERSONAL_CONFIDENTIAL.
    If the footer says CONFIDENTIAL but content is clearly HC, use HIGHLY_CONFIDENTIAL.
    Label reflects the correct classification, not the footer alone.

Add as many examples as you have — more is better, especially at boundaries:
    FOUO + PII   → PERSONAL_CONFIDENTIAL
    FOUO vs CONFIDENTIAL (impact threshold)
    CONFIDENTIAL vs HIGHLY_CONFIDENTIAL
"""

from data.generator import Sample, CLASS_TO_IDX

REAL_EXAMPLES: list[Sample] = [

    # ── Template — copy and fill in ───────────────────────────────────────────
    # Sample(
    #     text="""<paste extracted text here, anonymised>""",
    #     label="CONFIDENTIAL",           # one of the six class names
    #     label_idx=CLASS_TO_IDX["CONFIDENTIAL"],
    # ),

]
