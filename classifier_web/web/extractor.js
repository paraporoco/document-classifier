/**
 * extractor.js — Client-side document text extraction.
 *
 * Privacy guarantee: file bytes never leave the browser. All extraction
 * runs locally using browser APIs and CDN-loaded libraries.
 *
 * Supported formats (mirrors classifier_production/app.py):
 *   .pdf           pdf.js (text layer; image-only PDFs are not supported — no OCR)
 *   .docx          mammoth.js
 *   .xlsx          SheetJS
 *   .txt .md       FileReader (UTF-8 with latin-1 fallback)
 *   .html .htm     DOMParser (strips scripts/styles)
 *
 * Not supported in browser tier:
 *   .pptx          Returns status="unsupported_format" with an explanation
 *
 * Prerequisites (loaded as global scripts before this file):
 *   pdfjsLib       from pdfjs-dist (v3.x UMD build)
 *   mammoth        from mammoth.browser.min.js
 *   XLSX           from xlsx.full.min.js
 *
 * Public API (window.Extractor):
 *   Extractor.extract(file) → Promise<{ text, status, detail }>
 *     status: "ok" | "unsupported_format" | "no_text_layer" | "encrypted"
 *           | "oversized" | "extraction_failed"
 */

const Extractor = (() => {

  const MAX_BYTES = 10 * 1024 * 1024;  // 10 MB — matches CLI limit

  // ── Helpers ──────────────────────────────────────────────────────────────────

  function _err(status, detail) {
    return { text: null, status, detail };
  }

  function _ok(text) {
    return { text: text.trim(), status: "ok", detail: null };
  }

  /** Read a File as an ArrayBuffer. */
  function _readBuffer(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload  = () => resolve(r.result);
      r.onerror = () => reject(r.error);
      r.readAsArrayBuffer(file);
    });
  }

  /** Read a File as a UTF-8 string with latin-1 fallback. */
  async function _readText(file) {
    const buf = await _readBuffer(file);
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(buf);
    } catch {
      return new TextDecoder("latin-1").decode(buf);
    }
  }

  // ── Format extractors ────────────────────────────────────────────────────────

  async function _extractPdf(file) {
    if (typeof pdfjsLib === "undefined") {
      return _err("extraction_failed", "pdf.js not loaded.");
    }
    const buf = await _readBuffer(file);
    let pdf;
    try {
      const task = pdfjsLib.getDocument({ data: new Uint8Array(buf) });
      pdf = await task.promise;
    } catch (e) {
      if (e.name === "PasswordException") return _err("encrypted", "Password-protected PDF.");
      return _err("extraction_failed", `${e.name}: ${e.message}`);
    }

    const parts = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const page    = await pdf.getPage(i);
      const content = await page.getTextContent();
      const pageText = content.items.map(item => item.str).join(" ").trim();
      if (pageText) parts.push(pageText);
    }

    const text = parts.join("\n");
    if (text.length < 20) {
      return _err("no_text_layer",
        "No extractable text found. This may be an image-only PDF. " +
        "OCR is not available in the browser — use the desktop classify.exe instead.");
    }
    return _ok(text);
  }

  async function _extractDocx(file) {
    if (typeof mammoth === "undefined") {
      return _err("extraction_failed", "mammoth.js not loaded.");
    }
    const buf = await _readBuffer(file);
    try {
      const result = await mammoth.extractRawText({ arrayBuffer: buf });
      return _ok(result.value);
    } catch (e) {
      return _err("extraction_failed", `${e.name}: ${e.message}`);
    }
  }

  async function _extractXlsx(file) {
    if (typeof XLSX === "undefined") {
      return _err("extraction_failed", "SheetJS not loaded.");
    }
    const buf = await _readBuffer(file);
    try {
      const wb    = XLSX.read(buf, { type: "array" });
      const parts = [...wb.SheetNames];  // include sheet names as context
      for (const name of wb.SheetNames) {
        const ws   = wb.Sheets[name];
        const rows = XLSX.utils.sheet_to_json(ws, { header: 1, defval: "" });
        for (const row of rows) {
          const cells = row.map(c => String(c).trim()).filter(Boolean);
          if (cells.length) parts.push(cells.join(" "));
        }
      }
      return _ok(parts.join("\n"));
    } catch (e) {
      return _err("extraction_failed", `${e.name}: ${e.message}`);
    }
  }

  async function _extractHtml(file) {
    try {
      const raw  = await _readText(file);
      const doc  = new DOMParser().parseFromString(raw, "text/html");
      // Remove script and style elements
      doc.querySelectorAll("script, style").forEach(el => el.remove());
      const text = doc.body?.innerText ?? doc.documentElement.innerText ?? "";
      return _ok(text);
    } catch (e) {
      return _err("extraction_failed", `${e.name}: ${e.message}`);
    }
  }

  async function _extractText(file) {
    try {
      const text = await _readText(file);
      return _ok(text);
    } catch (e) {
      return _err("extraction_failed", `${e.name}: ${e.message}`);
    }
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Extract plain text from a File object.
   * @param {File} file
   * @returns {Promise<{text: string|null, status: string, detail: string|null}>}
   */
  async function extract(file) {
    if (file.size > MAX_BYTES) {
      return _err("oversized",
        `${(file.size / 1024 / 1024).toFixed(1)} MB exceeds the 10 MB limit.`);
    }

    const ext = file.name.split(".").pop().toLowerCase();

    try {
      switch (ext) {
        case "pdf":          return await _extractPdf(file);
        case "docx":         return await _extractDocx(file);
        case "xlsx":         return await _extractXlsx(file);
        case "html": case "htm": return await _extractHtml(file);
        case "txt": case "md":   return await _extractText(file);
        case "pptx":
          return _err("unsupported_format",
            ".pptx is not supported in the browser. " +
            "Use the desktop classify.exe for PowerPoint files.");
        default:
          return _err("unsupported_format", `'.${ext}' is not a supported format.`);
      }
    } catch (e) {
      return _err("extraction_failed", `${e.name}: ${e.message}`);
    }
  }

  return { extract };
})();

window.Extractor = Extractor;
