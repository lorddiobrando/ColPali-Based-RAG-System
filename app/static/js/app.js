/* ═══════════════════════════════════════════════════════
   ColPali RAG — Frontend Application Logic
   ═══════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ─── DOM References ───
  const form = document.getElementById("query-form");
  const input = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  const modeSelect = document.getElementById("mode-select");
  const topkSelect = document.getElementById("topk-select");

  const answerSection = document.getElementById("answer-section");
  const answerText = document.getElementById("answer-text");
  const answerModeBadge = document.getElementById("answer-mode-badge");
  const answerMeta = document.getElementById("answer-meta");

  const resultsSection = document.getElementById("results-section");
  const resultsGrid = document.getElementById("results-grid");
  const toggleDebugBtn = document.getElementById("toggle-debug");

  const debugSection = document.getElementById("debug-section");
  const debugGrid = document.getElementById("debug-grid");

  const skeletonSection = document.getElementById("skeleton-section");
  const emptyState = document.getElementById("empty-state");

  const badgeIndex = document.getElementById("badge-index");

  // ─── State ───
  let isLoading = false;
  let debugVisible = false;

  // ─── Helpers ───
  function showEl(el) { el.hidden = false; }
  function hideEl(el) { el.hidden = true; }

  function setLoading(on) {
    isLoading = on;
    submitBtn.disabled = on;
    if (on) {
      btnText.textContent = "Searching…";
      btnSpinner.hidden = false;
      hideEl(emptyState);
      hideEl(answerSection);
      hideEl(resultsSection);
      hideEl(debugSection);
      showEl(skeletonSection);
    } else {
      btnText.textContent = "Search";
      btnSpinner.hidden = true;
      hideEl(skeletonSection);
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ─── Render Answer ───
  function renderAnswer(generation) {
    if (!generation) { hideEl(answerSection); return; }

    if (generation.mode === "retrieval_only" || !generation.answer) {
      answerModeBadge.textContent = "Retrieval Only";
      answerModeBadge.className = "badge badge-amber";
      answerText.textContent = "Running in retrieval-only mode. Set OPENROUTER_API_KEY in .env to enable answer generation.";
    } else {
      answerModeBadge.textContent = "RAG";
      answerModeBadge.className = "badge badge-green";
      answerText.textContent = generation.answer;
    }

    const metaParts = [];
    if (generation.model) metaParts.push(`Model: ${generation.model}`);
    if (generation.timing) metaParts.push(`Generation: ${generation.timing.generation_seconds}s`);
    answerMeta.textContent = metaParts.join(" · ");

    showEl(answerSection);
  }

  // ─── Render Results ───
  function renderResults(results) {
    resultsGrid.innerHTML = "";
    if (!results || results.length === 0) {
      resultsGrid.innerHTML = '<p style="color:var(--muted);grid-column:1/-1;">No matching pages found.</p>';
      showEl(resultsSection);
      return;
    }

    results.forEach(function (r, idx) {
      const card = document.createElement("div");
      card.className = "result-card";

      const imgSrc = `/api/page-preview/${r.point_id}`;
      const docLabel = r.doc_id || "Unknown";
      const pageLabel = r.page_num != null ? `Page ${r.page_num}` : r.page_id || "—";
      const splitLabel = r.split || "";

      card.innerHTML = `
        <img class="result-card-image"
             src="${imgSrc}"
             alt="Page preview"
             title="Click to expand"
             loading="lazy"
             onclick="openLightbox('${imgSrc}', '${escapeHtml(docLabel)} - ${escapeHtml(pageLabel)}')"
             onerror="this.style.display='none'" />
        <div class="result-card-body">
          <div class="result-card-title">
            <span class="result-score">${r.score}</span>
            ${escapeHtml(pageLabel)}
          </div>
          <div class="result-card-meta">
            <strong>Doc:</strong> ${escapeHtml(docLabel)}<br/>
            <strong>Split:</strong> ${escapeHtml(splitLabel)}
          </div>
        </div>
      `;
      resultsGrid.appendChild(card);
    });

    showEl(resultsSection);
  }

  // ─── Render Debug ───
  function renderDebug(retrieval, generation) {
    debugGrid.innerHTML = "";

    const items = [];
    if (retrieval && retrieval.timing) {
      items.push({ label: "Init Time", value: `${retrieval.timing.init_seconds}s` });
      items.push({ label: "Retrieval Time", value: `${retrieval.timing.retrieval_seconds}s` });
      items.push({ label: "Total Retrieval", value: `${retrieval.timing.total_seconds}s` });
    }
    if (generation && generation.timing) {
      items.push({ label: "Generation Time", value: `${generation.timing.generation_seconds}s` });
    }
    if (retrieval) {
      items.push({ label: "Results Returned", value: `${(retrieval.results || []).length}` });
      items.push({ label: "Top-K Requested", value: `${retrieval.top_k}` });
    }
    if (generation) {
      items.push({ label: "Mode", value: generation.mode || "—" });
      items.push({ label: "Model", value: generation.model || "N/A" });
    }

    items.forEach(function (item) {
      const div = document.createElement("div");
      div.className = "debug-item";
      div.innerHTML = `
        <div class="debug-item-label">${escapeHtml(item.label)}</div>
        <div class="debug-item-value">${escapeHtml(item.value)}</div>
      `;
      debugGrid.appendChild(div);
    });

    if (debugVisible) showEl(debugSection);
  }

  // ─── Submit Handler ───
  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (isLoading) return;

    const query = input.value.trim();
    if (!query) return;

    const mode = modeSelect.value;
    const topK = parseInt(topkSelect.value, 10);

    setLoading(true);

    try {
      const resp = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: topK, mode: mode }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(function () { return {}; });
        throw new Error(errData.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();

      renderAnswer(data.generation);
      renderResults(data.retrieval ? data.retrieval.results : []);
      renderDebug(data.retrieval, data.generation);

      // Update index badge
      if (data.retrieval && data.retrieval.results) {
        badgeIndex.textContent = `Results: ${data.retrieval.results.length}`;
      }
    } catch (err) {
      hideEl(skeletonSection);
      answerModeBadge.textContent = "Error";
      answerModeBadge.className = "badge badge-amber";
      answerText.textContent = `Error: ${err.message}`;
      answerMeta.textContent = "";
      showEl(answerSection);
      hideEl(resultsSection);
    } finally {
      setLoading(false);
    }
  });

  // ─── Debug Toggle ───
  toggleDebugBtn.addEventListener("click", function () {
    debugVisible = !debugVisible;
    if (debugVisible) {
      showEl(debugSection);
      toggleDebugBtn.textContent = "Hide Debug";
    } else {
      hideEl(debugSection);
      toggleDebugBtn.textContent = "Debug Info";
    }
  });

  // ─── Lightbox ───
  const lightbox = document.createElement("div");
  lightbox.className = "lightbox";
  lightbox.innerHTML = `
    <span class="lightbox-close">&times;</span>
    <img class="lightbox-content" id="lightbox-img">
    <div class="lightbox-caption" id="lightbox-caption"></div>
  `;
  document.body.appendChild(lightbox);

  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxCaption = document.getElementById("lightbox-caption");
  const lightboxClose = lightbox.querySelector(".lightbox-close");

  lightboxClose.onclick = function() { lightbox.style.display = "none"; }
  lightbox.onclick = function(e) { if (e.target === lightbox) lightbox.style.display = "none"; }

  // Expose lightbox logic for dynamically rendered cards
  window.openLightbox = function(src, caption) {
    lightboxImg.src = src;
    lightboxCaption.textContent = caption;
    lightbox.style.display = "block";
  };

  // ─── Init ───
  console.log("ColPali RAG UI loaded.");
})();
