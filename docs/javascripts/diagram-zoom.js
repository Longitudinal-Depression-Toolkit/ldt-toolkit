(() => {
  const BOUND_ATTR = "data-mermaid-zoom-bound";
  let overlay;
  let overlayContent;
  let closeButton;
  let lastTrigger;
  let pendingScan = false;

  function ensureOverlay() {
    if (overlay) return;

    overlay = document.createElement("div");
    overlay.className = "mermaid-zoom-overlay";
    overlay.setAttribute("aria-hidden", "true");

    const panel = document.createElement("div");
    panel.className = "mermaid-zoom-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", "Expanded flowchart");

    closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "mermaid-zoom-close";
    closeButton.setAttribute("aria-label", "Close expanded flowchart");
    closeButton.textContent = "\u00D7";

    overlayContent = document.createElement("div");
    overlayContent.className = "mermaid-zoom-content";

    panel.appendChild(closeButton);
    panel.appendChild(overlayContent);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    closeButton.addEventListener("click", closeOverlay);
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        closeOverlay();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeOverlay();
      }
    });
  }

  function openOverlay(svgElement) {
    ensureOverlay();

    const clone = svgElement.cloneNode(true);
    clone.removeAttribute("width");
    clone.removeAttribute("height");

    overlayContent.innerHTML = "";
    overlayContent.appendChild(clone);

    overlay.classList.add("is-open");
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("mermaid-zoom-open");
    closeButton.focus();
  }

  function closeOverlay() {
    if (!overlay || !overlay.classList.contains("is-open")) return;

    overlay.classList.remove("is-open");
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("mermaid-zoom-open");

    window.setTimeout(() => {
      if (!overlay.classList.contains("is-open") && overlayContent) {
        overlayContent.innerHTML = "";
      }
    }, 150);

    if (lastTrigger && typeof lastTrigger.focus === "function") {
      lastTrigger.focus();
    }
  }

  function bindDiagram(svgElement) {
    if (svgElement.getAttribute(BOUND_ATTR) === "true") return;

    svgElement.setAttribute(BOUND_ATTR, "true");
    svgElement.setAttribute("tabindex", "0");
    svgElement.setAttribute("role", "button");
    svgElement.setAttribute("aria-label", "Open expanded flowchart");

    svgElement.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      lastTrigger = svgElement;
      openOverlay(svgElement);
    });

    svgElement.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      event.stopPropagation();
      lastTrigger = svgElement;
      openOverlay(svgElement);
    });
  }

  function initMermaidZoom(root = document) {
    ensureOverlay();
    const diagrams = root.querySelectorAll(
      ".md-typeset .mermaid svg, .md-typeset .mcs-stage-diagram svg"
    );
    diagrams.forEach(bindDiagram);
  }

  function scheduleScan() {
    if (pendingScan) return;
    pendingScan = true;
    window.requestAnimationFrame(() => {
      pendingScan = false;
      initMermaidZoom(document);
    });
  }

  function subscribeToNavigation() {
    if (typeof window.document$ !== "undefined" && typeof window.document$.subscribe === "function") {
      window.document$.subscribe(() => {
        window.setTimeout(() => initMermaidZoom(document), 0);
        window.requestAnimationFrame(() => initMermaidZoom(document));
      });
      return;
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => initMermaidZoom(document));
    } else {
      initMermaidZoom(document);
    }
  }

  subscribeToNavigation();

  const observer = new MutationObserver(() => {
    scheduleScan();
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  window.addEventListener("mcs:diagram-rendered", () => {
    scheduleScan();
  });
})();
