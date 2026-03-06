(() => {
  const BOUND_ATTR = "data-mcs-viz-bound";
  const RENDERED_ATTR = "data-stage-rendered";
  let mermaidApiPromise;
  let mermaidInitPromise;

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function loadMermaidApi() {
    if (
      typeof window.mermaid !== "undefined" &&
      typeof window.mermaid.render === "function"
    ) {
      return Promise.resolve(window.mermaid);
    }

    if (!mermaidApiPromise) {
      mermaidApiPromise = import(
        "https://unpkg.com/mermaid@10.4.0/dist/mermaid.esm.min.mjs"
      ).then((module) => module.default);
    }

    return mermaidApiPromise;
  }

  function ensureMermaidReady() {
    if (!mermaidInitPromise) {
      mermaidInitPromise = loadMermaidApi().then((mermaidApi) => {
        const isDark =
          document.body.getAttribute("data-md-color-scheme") === "slate";

        if (typeof mermaidApi.initialize === "function") {
          mermaidApi.initialize({
            startOnLoad: false,
            theme: isDark ? "dark" : "default",
          });
        }

        return mermaidApi;
      });
    }

    return mermaidInitPromise;
  }

  async function renderStageDiagram(node, stage) {
    const panel = node.querySelector(`[data-stage-panel="${stage}"]`);
    if (!panel || panel.getAttribute(RENDERED_ATTR) === "true") {
      return;
    }

    const target = panel.querySelector(`[data-stage-diagram="${stage}"]`);
    const sourceNode = panel.querySelector(`[data-stage-source="${stage}"]`);

    if (!target || !sourceNode) {
      return;
    }

    const source = (sourceNode.textContent || "").trim();
    if (!source) {
      target.innerHTML =
        '<p class="mcs-stage-diagram-error">No flowchart source available for this stage.</p>';
      panel.setAttribute(RENDERED_ATTR, "true");
      return;
    }

    try {
      const mermaidApi = await ensureMermaidReady();
      const renderId = `mcs_stage_${stage}_${Date.now()}_${Math.random()
        .toString(36)
        .slice(2, 8)}`;

      const rendered = await Promise.resolve(mermaidApi.render(renderId, source));
      if (!rendered || typeof rendered.svg !== "string") {
        throw new Error("Mermaid render result was empty.");
      }

      target.innerHTML = rendered.svg;
      if (typeof rendered.bindFunctions === "function") {
        rendered.bindFunctions(target);
      }

      panel.setAttribute(RENDERED_ATTR, "true");
    } catch (error) {
      const message =
        error && typeof error === "object" && "message" in error
          ? String(error.message)
          : String(error);

      target.innerHTML =
        `<pre class="mcs-stage-diagram-error">${escapeHtml(message)}</pre>`;
    }
  }

  function bindVisualization(node) {
    if (node.getAttribute(BOUND_ATTR) === "true") {
      return;
    }
    node.setAttribute(BOUND_ATTR, "true");

    const cards = Array.from(node.querySelectorAll("[data-stage-target]"));
    const panels = Array.from(node.querySelectorAll("[data-stage-panel]"));
    const hint = node.querySelector("[data-stage-hint]");

    if (!cards.length || !panels.length) {
      return;
    }

    let activeStage = "";

    const showStage = (stage) => {
      activeStage = stage;

      cards.forEach((card) => {
        const isActive = card.getAttribute("data-stage-target") === stage;
        card.classList.toggle("is-active", isActive);
        card.setAttribute("aria-pressed", isActive ? "true" : "false");
      });

      panels.forEach((panel) => {
        const isActive = panel.getAttribute("data-stage-panel") === stage;
        panel.hidden = !isActive;
      });

      if (hint) {
        hint.hidden = Boolean(stage);
      }

      if (stage) {
        renderStageDiagram(node, stage).then(() => {
          window.dispatchEvent(new CustomEvent("mcs:diagram-rendered"));
        });
      }
    };

    showStage("");

    cards.forEach((card) => {
      card.addEventListener("click", () => {
        const stage = card.getAttribute("data-stage-target") || "";
        showStage(activeStage === stage ? "" : stage);
      });
    });
  }

  function initVisualizations(root = document) {
    root.querySelectorAll("[data-mcs-viz]").forEach(bindVisualization);
  }

  function init() {
    initVisualizations(document);

    if (
      typeof window.document$ !== "undefined" &&
      typeof window.document$.subscribe === "function"
    ) {
      window.document$.subscribe(() => {
        window.requestAnimationFrame(() => initVisualizations(document));
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
