/**
 * app.js — SPA application logic for the document sensitivity classifier.
 *
 * Depends on (loaded before this file):
 *   window.Engine     (engine.js)
 *   window.Extractor  (extractor.js)
 *
 * Features:
 *   - Drag-and-drop + file picker
 *   - Text paste / direct text classification
 *   - Batch processing (sequential) with live results table
 *   - CSV export of results
 *   - Service Worker registration for artifact caching
 */

"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────

const MANIFEST_URL = "/web_artefacts/manifest.json";

const BADGE_COLOURS = {
  PUBLIC:                    { bg: "#22c55e", fg: "#fff" },
  FOUO:                      { bg: "#3b82f6", fg: "#fff" },
  CONFIDENTIAL:              { bg: "#f59e0b", fg: "#fff" },
  PERSONAL_CONFIDENTIAL:     { bg: "#f97316", fg: "#fff" },
  HIGHLY_CONFIDENTIAL:       { bg: "#ef4444", fg: "#fff" },
  PERSONAL_HIGHLY_CONFIDENTIAL: { bg: "#7f1d1d", fg: "#fff" },
};

// ── DOM refs (populated in init) ─────────────────────────────────────────────

let ui = {};

// ── State ────────────────────────────────────────────────────────────────────

const results = [];   // { filename, classification, confidence, review_flag, reasoning, policy_version, timestamp, status, detail }

// ── Helpers ──────────────────────────────────────────────────────────────────

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmtConf(conf) {
  return typeof conf === "number" ? (conf * 100).toFixed(1) + "%" : "—";
}

function now() {
  return new Date().toISOString().replace("T", " ").slice(0, 19) + "Z";
}

// ── Loading screen ────────────────────────────────────────────────────────────

function showLoader(phase, done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const labels = { manifest: "Manifest", model: "Model", vectorizer: "Vocabulary", temperature: "Calibration" };
  ui.loadStatus.textContent = `Downloading ${labels[phase] ?? phase}…`;
  ui.loadBar.style.width    = pct + "%";
  ui.loadPct.textContent    = pct + "%";
}

function showReady(fromCache) {
  ui.loader.hidden = true;
  ui.main.hidden   = false;
  ui.cacheNote.hidden   = !fromCache;
  ui.cacheNote.textContent = fromCache ? "Model loaded from cache." : "";
}

// ── Badge ─────────────────────────────────────────────────────────────────────

function badge(label) {
  const c = BADGE_COLOURS[label] ?? { bg: "#6b7280", fg: "#fff" };
  return `<span class="badge" style="background:${c.bg};color:${c.fg}">${esc(label)}</span>`;
}

function confBar(conf) {
  const pct = Math.round(conf * 100);
  const col = conf >= 0.9 ? "#22c55e" : conf >= 0.75 ? "#f59e0b" : "#ef4444";
  return `<div class="conf-bar-wrap"><div class="conf-bar" style="width:${pct}%;background:${col}"></div><span>${pct}%</span></div>`;
}

// ── Results table ─────────────────────────────────────────────────────────────

function renderTable() {
  if (results.length === 0) {
    ui.resultsSection.hidden = true;
    return;
  }
  ui.resultsSection.hidden = false;

  const rows = results.map(r => {
    if (r.status !== "ok") {
      return `<tr class="row-error">
        <td>${esc(r.filename)}</td>
        <td colspan="4"><span class="err-tag">${esc(r.status)}</span> ${esc(r.detail ?? "")}</td>
      </tr>`;
    }
    const review = r.review_flag
      ? `<span class="review-flag" title="Low confidence — manual review recommended">⚑ Review</span>`
      : "";
    return `<tr>
      <td title="${esc(r.filename)}">${esc(r.filename.length > 40 ? r.filename.slice(0, 37) + "…" : r.filename)}</td>
      <td>${badge(r.classification)}</td>
      <td>${confBar(r.confidence)} ${review}</td>
      <td class="reasoning-cell" title="${esc(r.reasoning)}">${esc(r.reasoning)}</td>
      <td>${esc(r.timestamp)}</td>
    </tr>`;
  }).join("");

  ui.tableBody.innerHTML = rows;
}

// ── Single-result card (for text-paste mode) ───────────────────────────────────

function renderCard(result) {
  const review = result.review_flag
    ? `<div class="alert-review">⚑ Confidence below threshold — manual review recommended.</div>`
    : "";
  ui.card.innerHTML = `
    <div class="card-label">${badge(result.classification)}</div>
    <div class="card-conf">Confidence: <strong>${fmtConf(result.confidence)}</strong></div>
    ${review}
    <div class="card-reasoning">${esc(result.reasoning)}</div>
    <div class="card-meta">Policy v${esc(result.policy_version)}</div>
  `;
  ui.card.hidden = false;
}

// ── Processing ────────────────────────────────────────────────────────────────

async function classifyText(text, filename) {
  const result = await Engine.classify(text);
  result.filename  = filename;
  result.timestamp = now();
  result.status    = "ok";
  result.detail    = null;
  return result;
}

async function processFile(file) {
  const { text, status, detail } = await Extractor.extract(file);
  if (status !== "ok") {
    return { filename: file.name, status, detail, timestamp: now() };
  }
  return classifyText(text, file.name);
}

async function processFiles(files) {
  ui.progressWrap.hidden = false;
  ui.progressText.textContent = `Processing 0 / ${files.length}…`;

  for (let i = 0; i < files.length; i++) {
    ui.progressText.textContent = `Processing ${i + 1} / ${files.length}: ${files[i].name}`;
    try {
      const result = await processFile(files[i]);
      results.unshift(result);
    } catch (e) {
      results.unshift({ filename: files[i].name, status: "extraction_failed", detail: e.message, timestamp: now() });
    }
    renderTable();
  }

  ui.progressWrap.hidden = true;
  ui.progressText.textContent = "";
}

// ── CSV export ────────────────────────────────────────────────────────────────

function exportCsv() {
  const COLS = ["timestamp", "filename", "classification", "confidence", "review_flag", "reasoning", "policy_version", "status", "detail"];
  const lines = [COLS.join(",")];
  for (const r of results) {
    lines.push(COLS.map(k => {
      const v = r[k] ?? "";
      const s = String(v);
      return s.includes(",") || s.includes('"') || s.includes("\n")
        ? '"' + s.replace(/"/g, '""') + '"'
        : s;
    }).join(","));
  }
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = `classification_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Event wiring ──────────────────────────────────────────────────────────────

function wireEvents() {
  // File picker
  ui.fileInput.addEventListener("change", async () => {
    const files = Array.from(ui.fileInput.files);
    if (files.length) await processFiles(files);
    ui.fileInput.value = "";
  });

  // Drop zone
  ui.dropZone.addEventListener("dragover", e => { e.preventDefault(); ui.dropZone.classList.add("drag-over"); });
  ui.dropZone.addEventListener("dragleave", ()  => ui.dropZone.classList.remove("drag-over"));
  ui.dropZone.addEventListener("drop", async e  => {
    e.preventDefault();
    ui.dropZone.classList.remove("drag-over");
    const files = Array.from(e.dataTransfer.files);
    if (files.length) await processFiles(files);
  });
  ui.dropZone.addEventListener("click", () => ui.fileInput.click());

  // Text paste classify
  ui.classifyTextBtn.addEventListener("click", async () => {
    const text = ui.textInput.value.trim();
    if (!text) { ui.textInput.focus(); return; }
    ui.classifyTextBtn.disabled = true;
    ui.card.hidden = true;
    try {
      const result = await Engine.classify(text);
      result.filename  = "(pasted text)";
      result.timestamp = now();
      result.status    = "ok";
      renderCard(result);
      results.unshift({ ...result, detail: null });
      renderTable();
    } catch (e) {
      ui.card.innerHTML = `<div class="alert-review">Error: ${esc(e.message)}</div>`;
      ui.card.hidden = false;
    } finally {
      ui.classifyTextBtn.disabled = false;
    }
  });

  // Clear text on Ctrl+Enter
  ui.textInput.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ui.classifyTextBtn.click();
  });

  // CSV export
  ui.exportBtn.addEventListener("click", exportCsv);

  // Clear results
  ui.clearBtn.addEventListener("click", () => {
    results.length = 0;
    ui.tableBody.innerHTML = "";
    ui.resultsSection.hidden = true;
    ui.card.hidden = true;
  });
}

// ── Service Worker registration ───────────────────────────────────────────────

async function registerSW() {
  if (!("serviceWorker" in navigator)) return;
  try {
    await navigator.serviceWorker.register("/sw.js");
  } catch (e) {
    console.warn("Service Worker registration failed:", e);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  ui = {
    loader:          document.getElementById("loader"),
    main:            document.getElementById("main"),
    loadStatus:      document.getElementById("load-status"),
    loadBar:         document.getElementById("load-bar"),
    loadPct:         document.getElementById("load-pct"),
    cacheNote:       document.getElementById("cache-note"),
    dropZone:        document.getElementById("drop-zone"),
    fileInput:       document.getElementById("file-input"),
    textInput:       document.getElementById("text-input"),
    classifyTextBtn: document.getElementById("classify-text-btn"),
    card:            document.getElementById("result-card"),
    resultsSection:  document.getElementById("results-section"),
    tableBody:       document.getElementById("results-body"),
    exportBtn:       document.getElementById("export-btn"),
    clearBtn:        document.getElementById("clear-btn"),
    progressWrap:    document.getElementById("progress-wrap"),
    progressText:    document.getElementById("progress-text"),
  };

  wireEvents();

  let fromCache = false;
  try {
    // Check if SW already has the artefacts cached
    if ("caches" in window) {
      const cache = await caches.open("dc-artefacts-v1");
      const keys  = await cache.keys();
      fromCache    = keys.length > 0;
    }
  } catch { /* ignore */ }

  try {
    await Engine.load(MANIFEST_URL, e => showLoader(e.phase, e.done, e.total));
    showReady(fromCache);
  } catch (e) {
    ui.loadStatus.textContent = `Failed to load model: ${e.message}`;
    ui.loadBar.style.background = "#ef4444";
    return;
  }

  registerSW();
}

document.addEventListener("DOMContentLoaded", init);
