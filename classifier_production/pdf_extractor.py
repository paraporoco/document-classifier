from io import StringIO
from pathlib import Path
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.pdfdocument import PDFPasswordIncorrect

MIN_CHARS = 20

class ExtractionError(Exception):
    pass

def extract(path: Path) -> tuple[str, str]:
    buf = StringIO()
    try:
        with open(path, "rb") as f:
            extract_text_to_fp(f, buf, laparams=LAParams(), output_type="text", codec="utf-8")
    except PDFPasswordIncorrect:
        return "", "encrypted"
    except PDFSyntaxError as e:
        raise ExtractionError(f"PDF parse error: {e}") from e
    except Exception as e:
        raise ExtractionError(f"PDF error: {e}") from e

    text = buf.getvalue().strip()
    return (text, "ok") if len(text) >= MIN_CHARS else (text, "no_text_layer")
