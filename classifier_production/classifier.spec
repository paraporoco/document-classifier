# classifier.spec
# Run from the classifier\ folder:
#   pyinstaller classifier.spec

block_cipher = None

from PyInstaller.utils.hooks import collect_all, collect_submodules

# pymupdf and PIL need full collection — their C extensions aren't auto-detected
pymupdf_datas, pymupdf_binaries, pymupdf_hiddenimports = collect_all("pymupdf")
fitz_datas,    fitz_binaries,    fitz_hiddenimports    = collect_all("fitz")
pil_datas,     pil_binaries,     pil_hiddenimports     = collect_all("PIL")
# pymupdf 1.25+ ships a separate pymupdf_fonts package
try:
    fonts_datas, fonts_binaries, fonts_hiddenimports = collect_all("pymupdf_fonts")
except Exception:
    fonts_datas, fonts_binaries, fonts_hiddenimports = [], [], []

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=pymupdf_binaries + fitz_binaries + pil_binaries + fonts_binaries,
    datas=[
        ('artefacts/model_ssl.pkl',   'artefacts'),
        ('artefacts/vectorizer.pkl',  'artefacts'),
        ('artefacts/temperature.pkl', 'artefacts'),
    ] + pymupdf_datas + fitz_datas + pil_datas + fonts_datas,
    hiddenimports=(
        pymupdf_hiddenimports + fitz_hiddenimports + pil_hiddenimports + fonts_hiddenimports + [
        # sklearn — core
        'sklearn',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.utils.validation',
        'sklearn.utils.multiclass',
        'sklearn.neighbors._partition_nodes',
        # sklearn — neural network (the model)
        'sklearn.neural_network',
        'sklearn.neural_network._multilayer_perceptron',
        'sklearn.neural_network._rbm',
        # sklearn — feature extraction (the vectoriser)
        'sklearn.feature_extraction',
        'sklearn.feature_extraction.text',
        'sklearn.feature_extraction._stop_words',
        # sklearn — preprocessing
        'sklearn.preprocessing',
        'sklearn.preprocessing._encoders',
        'sklearn.preprocessing._data',
        # sklearn — misc
        'sklearn.pipeline',
        'sklearn.linear_model',
        'sklearn.metrics',
        # pdf
        'pypdf',
        'pypdf._reader',
        'pypdf._page',
        'pypdf.generic',
        'pdfminer.high_level',
        'pdfminer.layout',
        'pdfminer.pdfpage',
        'pdfminer.pdfinterp',
        'pdfminer.pdfdevice',
        'pdfminer.converter',
        'pdfminer.pdfdocument',
        'pdfminer.pdfparser',
        # ocr
        'pytesseract',
        # office
        'docx',
        'openpyxl',
        'pptx',
    ]),
    excludes=['matplotlib','tkinter','pytest','torch','tensorflow'],
    hookspath=[],
    runtime_hooks=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='classify',
    console=True,
    upx=True,
)
