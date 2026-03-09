/**
 * sw.js — Service Worker for the document sensitivity classifier.
 *
 * Strategy:
 *   manifest.json       network-first (always revalidate; controls which artefacts to cache)
 *   *.onnx / *.json     cache-first (content-hashed filenames → safe to cache forever)
 *   everything else     network-first with cache fallback
 *
 * On install: fetch manifest, cache all three artefacts.
 * On activate: delete cached artefact files that are no longer in the manifest
 *              (handles the case where a retrain produces new content hashes).
 */

const ARTEFACTS_CACHE = "dc-artefacts-v1";
const MANIFEST_URL    = "/web_artefacts/manifest.json";

// ── Install ───────────────────────────────────────────────────────────────────
// Pre-cache artefacts listed in the manifest.

self.addEventListener("install", event => {
  event.waitUntil(
    (async () => {
      const manifest   = await fetch(MANIFEST_URL, { cache: "no-store" }).then(r => r.json());
      const artefactUrls = [
        "/web_artefacts/" + manifest.model,
        "/web_artefacts/" + manifest.vectorizer,
        "/web_artefacts/" + manifest.temperature,
      ];
      const cache = await caches.open(ARTEFACTS_CACHE);
      await cache.addAll(artefactUrls);
      // Skip waiting so the new SW activates immediately on first install
      self.skipWaiting();
    })().catch(e => console.warn("[SW] Install failed:", e))
  );
});

// ── Activate ──────────────────────────────────────────────────────────────────
// Purge artefact files that are no longer referenced by the current manifest.

self.addEventListener("activate", event => {
  event.waitUntil(
    (async () => {
      await clients.claim();

      let currentFiles;
      try {
        const manifest = await fetch(MANIFEST_URL, { cache: "no-store" }).then(r => r.json());
        currentFiles = new Set([
          "/web_artefacts/" + manifest.model,
          "/web_artefacts/" + manifest.vectorizer,
          "/web_artefacts/" + manifest.temperature,
        ]);
      } catch {
        return;  // If manifest is unreachable, leave the cache untouched
      }

      const cache = await caches.open(ARTEFACTS_CACHE);
      const keys  = await cache.keys();
      for (const req of keys) {
        const url = new URL(req.url).pathname;
        if (url.startsWith("/web_artefacts/") && url !== "/web_artefacts/manifest.json") {
          if (!currentFiles.has(url)) {
            await cache.delete(req);
          }
        }
      }
    })()
  );
});

// ── Fetch ─────────────────────────────────────────────────────────────────────

self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);

  if (url.pathname === "/web_artefacts/manifest.json") {
    // Network-first: always revalidate the manifest
    event.respondWith(
      fetch(event.request)
        .then(r => {
          caches.open(ARTEFACTS_CACHE).then(c => c.put(event.request, r.clone()));
          return r;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  if (url.pathname.startsWith("/web_artefacts/")) {
    // Cache-first: content-hashed files never change
    event.respondWith(
      caches.open(ARTEFACTS_CACHE).then(async cache => {
        const cached = await cache.match(event.request);
        if (cached) return cached;
        const response = await fetch(event.request);
        if (response.ok) cache.put(event.request, response.clone());
        return response;
      })
    );
    return;
  }

  // All other requests: network-first with cache fallback
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
