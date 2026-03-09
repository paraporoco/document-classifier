# Document Classifier

A self-contained Windows CLI tool that classifies documents by data sensitivity level.

## Classification levels

| Level | Description |
|---|---|
| `PUBLIC` | No restriction. Intended for general audiences. |
| `INTERNAL` | Internal use only. Not for external distribution. |
| `CONFIDENTIAL` | Business-sensitive. Restricted access. |
| `PRIVATE_CONFIDENTIAL` | Confidential + personal/individual data. |
| `HIGHLY_CONFIDENTIAL` | Highest organisational sensitivity. |
| `PRIVATE_HIGHLY_CONFIDENTIAL` | Highly confidential + personal data. |

## Supported formats

`.txt` `.md` `.pdf` `.html` `.htm` `.docx` `.xlsx` `.pptx`

PDF files with no text layer (scanned, screenshots) are processed via OCR if Tesseract is installed.

## Usage

```bat
REM Classify a folder
classify.exe C:\Documents --output results.csv

REM Classify a single file
classify.exe C:\Documents\contract.pdf

REM Recursive, with OCR for scanned PDFs
classify.exe C:\Documents --recursive --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"

REM Show only files flagged for human review (confidence < 0.75)
classify.exe C:\Documents --review-only
```

## Options

| Option | Default | Description |
|---|---|---|
| `input` | required | File or folder to classify |
| `--output` | `classification_log.csv` | Output CSV path |
| `--recursive` | off | Walk subdirectories |
| `--review-only` | off | Log only low-confidence results |
| `--min-confidence` | 0.0 | Skip files below this confidence score |
| `--tesseract` | auto | Path to `tesseract.exe` for OCR on image PDFs |

## Output files

| File | Contents |
|---|---|
| `classification_log.csv` | Full results — one row per file |
| `classification_log_slim.csv` | Filename + classification only |
| `classification_log_summary.txt` | Run summary with counts per level |
| `classification_log_review.csv` | Files flagged for human review (if any) |

## Building the executable

```bat
pip install scikit-learn numpy scipy pdfminer.six python-docx openpyxl python-pptx pypdf pymupdf Pillow pytesseract pyinstaller
pyinstaller classifier.spec
```

Output: `dist\classify.exe` (~150 MB, fully self-contained)

## OCR setup (optional)

For scanned PDFs, install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).  
Default detection path: `C:\Program Files\Tesseract-OCR\tesseract.exe`  
Use `--tesseract` to override.

## Changing the classification hierarchy

See `app.py` (`CLASSES` constant and `_summary()`) and `classifier/data/generator.py`.  
A hierarchy change requires retraining the model and replacing the `.pkl` files in `artefacts/`.
