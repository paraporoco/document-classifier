/**
 * engine.js — Client-side inference engine for the document sensitivity classifier.
 *
 * Prerequisites (loaded as global scripts before this file):
 *   onnxruntime-web (exposes window.ort)
 *
 * Public API (exposed as window.Engine):
 *   Engine.load(manifestUrl, onProgress?) → Promise<void>
 *     Download and cache artefacts. onProgress(e) is called with:
 *       { phase: "manifest"|"model"|"vectorizer"|"temperature", done: n, total: 3 }
 *
 *   Engine.classify(text) → Promise<Result>
 *     Result: { classification, confidence, review_flag, reasoning, policy_version }
 *
 *   Engine.isReady() → bool
 *   Engine.classes   → string[] (the six sensitivity labels)
 *
 * Privacy guarantee: no text or document content is ever sent over the network.
 * All inference executes locally in WebAssembly via onnxruntime-web.
 *
 * Inference pipeline (mirrors classifier_production/predict.py exactly):
 *   1. Tokenise: unicode-normalise → lowercase → regex unigrams + bigrams
 *   2. TF-IDF: sublinear TF × IDF, L2-normalise
 *   3. ONNX MLP: predict_proba via onnxruntime-web
 *   4. Temperature calibration: log → scale by 1/T → softmax
 *   5. Reasoning: top-5 features by (tfidf_value × coef_weight) contribution
 */

const Engine = (() => {
  // ── State ────────────────────────────────────────────────────────────────────
  let _session  = null;   // ort.InferenceSession
  let _vec      = null;   // { vocabulary, idf, coef_weights, ... }
  let _T        = null;   // temperature scalar (float)
  let _manifest = null;   // parsed manifest.json
  let _ready    = false;

  // ── Tokenisation ─────────────────────────────────────────────────────────────
  //
  // Mirrors sklearn TfidfVectorizer(strip_accents="unicode", lowercase=True,
  //   ngram_range=(1,2), token_pattern=r"(?u)\b\w\w+\b")
  //
  // Note: Python's (?u)\w matches Unicode word characters (letters from any
  // script). JS \w matches ASCII [a-zA-Z0-9_] only, even with the /u flag.
  // After strip_accents("unicode") most Latin-script text becomes ASCII, so
  // parity is very high in practice. Non-Latin script tokens (Chinese, Arabic,
  // etc.) will be missed by the JS tokeniser — acceptable for the current corpus.

  function _stripAccents(str) {
    // NFD decompose then strip combining diacritical marks (Unicode category Mn)
    return str.normalize("NFD").replace(/[\u0300-\u036f\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]/g, "");
  }

  function _tokenise(text) {
    const normalised = _stripAccents(text).toLowerCase();
    // Match 2+ word-character runs (mirrors sklearn's default token_pattern)
    const unigrams = normalised.match(/\b\w\w+\b/g) || [];
    // Append bigrams
    const tokens = unigrams.slice();
    for (let i = 0; i < unigrams.length - 1; i++) {
      tokens.push(unigrams[i] + " " + unigrams[i + 1]);
    }
    return tokens;
  }

  // ── TF-IDF ───────────────────────────────────────────────────────────────────
  //
  // sublinear_tf=True : tf(t,d) = 1 + ln(count)  if count > 0, else 0
  // idf               : from vectorizer.json (sklearn smooth IDF)
  // norm="l2"         : L2-normalise the document vector

  function _tfIdf(text) {
    const vocab  = _vec.vocabulary;
    const idf    = _vec.idf;
    const n      = idf.length;
    const vec    = new Float32Array(n);

    // Count term occurrences for tokens present in vocabulary
    const counts = Object.create(null);
    for (const token of _tokenise(text)) {
      if (token in vocab) {
        counts[token] = (counts[token] || 0) + 1;
      }
    }

    // Sublinear TF × IDF
    for (const [token, count] of Object.entries(counts)) {
      const idx = vocab[token];
      vec[idx] = (1.0 + Math.log(count)) * idf[idx];  // Math.log = natural log
    }

    // L2 normalise
    let norm = 0.0;
    for (let i = 0; i < n; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0.0) {
      for (let i = 0; i < n; i++) vec[i] /= norm;
    }

    return vec;  // Float32Array of length n_features
  }

  // ── Temperature calibration ──────────────────────────────────────────────────
  //
  // Identical to the chain in predict.py:
  //   logits = log(proba + 1e-12)
  //   scaled = logits / T
  //   proba_cal = softmax(scaled)

  function _temperatureScale(proba) {
    const n      = proba.length;
    const scaled = new Float64Array(n);
    let   maxVal = -Infinity;

    for (let i = 0; i < n; i++) {
      scaled[i] = Math.log(proba[i] + 1e-12) / _T;
      if (scaled[i] > maxVal) maxVal = scaled[i];
    }

    let sum = 0.0;
    const exp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      exp[i] = Math.exp(scaled[i] - maxVal);
      sum += exp[i];
    }

    return Array.from(exp, e => e / sum);
  }

  // ── Reasoning ────────────────────────────────────────────────────────────────
  //
  // Mirrors _reasoning() in predict.py:
  //   contribution[i] = featureVec[i] * |coefs_[0]|.sum(axis=1)[i]
  //   top-5 by contribution, excluding zeros

  function _topFeatures(featureVec) {
    const weights = _vec.coef_weights;
    const vocab   = _vec.vocabulary;

    // Build reverse map: feature index → token (first token that maps there)
    if (!_vec._indexToToken) {
      const rev = {};
      for (const [token, idx] of Object.entries(vocab)) {
        if (!(idx in rev)) rev[idx] = token;
      }
      _vec._indexToToken = rev;
    }
    const rev = _vec._indexToToken;

    // Score and rank
    const scored = [];
    for (let i = 0; i < featureVec.length; i++) {
      const s = featureVec[i] * weights[i];
      if (s > 0) scored.push({ i, s });
    }
    scored.sort((a, b) => b.s - a.s);

    const top = scored.slice(0, 5);
    if (top.length === 0) return "No strong signals.";
    return "Top signals: " + top.map(({ i, s }) => `${rev[i]} (${s.toFixed(3)})`).join(", ");
  }

  // ── Artefact fetching ────────────────────────────────────────────────────────

  async function _fetchJson(url) {
    const r = await fetch(url, { cache: "default" });
    if (!r.ok) throw new Error(`Fetch failed ${url} (${r.status})`);
    return r.json();
  }

  async function _fetchBinary(url) {
    const r = await fetch(url, { cache: "default" });
    if (!r.ok) throw new Error(`Fetch failed ${url} (${r.status})`);
    return r.arrayBuffer();
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Download and initialise all artefacts.
   * @param {string} manifestUrl  URL to manifest.json (e.g. "/web_artefacts/manifest.json")
   * @param {Function} [onProgress]  Called after each artefact loads: ({phase, done, total})
   */
  async function load(manifestUrl, onProgress) {
    _ready = false;

    // 1. Manifest (small; always re-fetched to detect updates)
    onProgress?.({ phase: "manifest", done: 0, total: 3 });
    _manifest = await _fetchJson(manifestUrl);
    const base = manifestUrl.replace(/manifest\.json$/, "");

    // 2. Fetch model, vectorizer, temperature in parallel.
    // Use a shared counter so progress reflects completion order, not artifact order.
    // (The ONNX model is the largest file and typically finishes last; fixed ordinals
    // would leave the bar at 33% when the model completes after the two JSON files.)
    let _done = 0;
    const [onnxBuf, vecData, tempData] = await Promise.all([
      _fetchBinary(base + _manifest.model).then(b  => { onProgress?.({ phase: "model",       done: ++_done, total: 3 }); return b;  }),
      _fetchJson(base + _manifest.vectorizer).then(d => { onProgress?.({ phase: "vectorizer",  done: ++_done, total: 3 }); return d;  }),
      _fetchJson(base + _manifest.temperature).then(d => { onProgress?.({ phase: "temperature", done: ++_done, total: 3 }); return d;  }),
    ]);

    // 3. Create ONNX inference session
    _session = await ort.InferenceSession.create(onnxBuf, {
      executionProviders: ["wasm"],
    });

    _vec   = vecData;
    _T     = tempData.T;
    _ready = true;
  }

  /**
   * Classify a plain-text string.
   * Returns { classification, confidence, review_flag, reasoning, policy_version }
   */
  async function classify(text) {
    if (!_ready) throw new Error("Engine not ready — call Engine.load() first.");
    if (typeof text !== "string" || !text.trim()) {
      throw new Error("text must be a non-empty string.");
    }

    // Step 1: TF-IDF feature vector
    const featureVec = _tfIdf(text);

    // Step 2: ONNX inference
    //   skl2onnx MLPClassifier with zipmap=False produces two outputs:
    //     output[0] "label"              — int64 [1]          (predicted class index)
    //     output[1] "output_probability" — float32 [1, n_cls] (predict_proba)
    const tensor  = new ort.Tensor("float32", featureVec, [1, featureVec.length]);
    const results = await _session.run({ float_input: tensor });

    // Extract flat probability array
    const probKey  = Object.keys(results).find(k => results[k].dims?.length === 2)
                  ?? Object.keys(results)[1];
    const proba    = Array.from(results[probKey].data);

    // Step 3: Temperature calibration
    const calibrated = _temperatureScale(proba);

    // Step 4: Argmax
    const idx        = calibrated.reduce((best, p, i) => p > calibrated[best] ? i : best, 0);
    const confidence = Math.round(calibrated[idx] * 10000) / 10000;

    return {
      classification: _manifest.classes[idx],
      confidence,
      review_flag:    confidence < _manifest.review_threshold,
      reasoning:      _topFeatures(featureVec),
      policy_version: _manifest.policy_version,
    };
  }

  return {
    load,
    classify,
    isReady:  ()  => _ready,
    get classes()       { return _manifest?.classes ?? []; },
    get policyVersion() { return _manifest?.policy_version ?? null; },
  };
})();

window.Engine = Engine;
