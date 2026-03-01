(() => {
  const BOUND_ATTR = "data-image-zoom-bound";
  let overlay;
  let overlayImage;

  function ensureOverlay() {
    if (overlay) return;

    overlay = document.createElement("div");
    overlay.className = "image-zoom-overlay";
    overlay.setAttribute("aria-hidden", "true");

    overlayImage = document.createElement("img");
    overlayImage.className = "image-zoom-overlay__image";
    overlayImage.alt = "";

    overlay.appendChild(overlayImage);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", closeOverlay);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeOverlay();
      }
    });
  }

  function openOverlay(source, altText) {
    ensureOverlay();
    overlayImage.src = source;
    overlayImage.alt = altText || "";
    overlay.classList.add("is-open");
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("image-zoom-open");
  }

  function closeOverlay() {
    if (!overlay || !overlay.classList.contains("is-open")) return;
    overlay.classList.remove("is-open");
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("image-zoom-open");
    window.setTimeout(() => {
      if (!overlay.classList.contains("is-open")) {
        overlayImage.removeAttribute("src");
      }
    }, 180);
  }

  function bindImage(image) {
    if (image.getAttribute(BOUND_ATTR) === "true") return;
    image.setAttribute(BOUND_ATTR, "true");
    image.classList.add("zoomable-image");

    image.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      openOverlay(image.currentSrc || image.src, image.alt);
    });
  }

  function initImageZoom(root = document) {
    ensureOverlay();
    const images = root.querySelectorAll(".md-content .md-typeset img");
    images.forEach((image) => {
      if (image.closest(".no-image-zoom")) return;
      bindImage(image);
    });
  }

  if (typeof window.document$ !== "undefined" && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => initImageZoom(document));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initImageZoom(document));
  } else {
    initImageZoom(document);
  }
})();
