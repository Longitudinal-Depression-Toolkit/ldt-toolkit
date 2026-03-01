(() => {
  const EXCLUDED_ROUTE_SUFFIXES = new Set([
    "/community-hub",
    "/getting-started/overview",
    "/tutorials",
    "/api",
    "/study-reproducibility",
  ]);
  const MARKDOWN_ACTIONS = new Set([
    "copy-markdown-link",
    "view-markdown",
    "open-chatgpt",
    "open-claude",
  ]);
  let resolvedMdUrlCache = null;

  function normalizePath(pathname) {
    let path = pathname || "/";
    path = path.split("#")[0].split("?")[0];
    if (path.endsWith("/index.html")) {
      path = path.slice(0, -"/index.html".length);
    } else if (path.endsWith(".html")) {
      path = path.slice(0, -".html".length);
    }
    path = path.replace(/\/+$/, "");
    return path || "/";
  }

  function getMetaContent(name) {
    const node = document.querySelector(`meta[name="${name}"]`);
    return node ? node.content.trim() : "";
  }

  function stripReadTheDocsPrefix(path) {
    const rtdData = window.READTHEDOCS_DATA;
    if (rtdData && typeof rtdData === "object") {
      const language = typeof rtdData.language === "string"
        ? rtdData.language.replace(/^\/+|\/+$/g, "")
        : "";
      const version = typeof rtdData.version === "string"
        ? rtdData.version.replace(/^\/+|\/+$/g, "")
        : "";
      if (language && version) {
        const prefix = `/${language}/${version}`;
        if (path === prefix) return "/";
        if (path.startsWith(`${prefix}/`)) {
          return path.slice(prefix.length) || "/";
        }
      }
    }

    const host = window.location.hostname.toLowerCase();
    const isReadTheDocsHost = host === "readthedocs.io" || host.endsWith(".readthedocs.io");
    if (!isReadTheDocsHost) return path;

    const segments = path.replace(/^\/+|\/+$/g, "").split("/");
    if (segments.length < 3) return path;
    if (!/^[a-z]{2}(?:-[a-z]{2})?$/i.test(segments[0])) return path;
    return `/${segments.slice(2).join("/")}`;
  }

  function stripBasePath(path) {
    const pathWithoutReadTheDocsPrefix = stripReadTheDocsPrefix(path);
    const configuredBase = getMetaContent("mkdocs-copy-to-llm-base-path");
    const prefixes = [];

    if (configuredBase) {
      prefixes.push(configuredBase);
    }

    // Common deployment prefixes for this repo.
    prefixes.push("/Longitudinal-Depression-Toolkit/ldt-toolkit");
    prefixes.push("/ldt-toolkit");

    for (const candidate of prefixes) {
      const base = candidate.startsWith("/") ? candidate : `/${candidate}`;
      const normalizedBase = base.replace(/\/+$/, "");
      if (!normalizedBase || normalizedBase === "/") continue;
      if (pathWithoutReadTheDocsPrefix === normalizedBase) return "/";
      if (pathWithoutReadTheDocsPrefix.startsWith(`${normalizedBase}/`)) {
        return pathWithoutReadTheDocsPrefix.slice(normalizedBase.length) || "/";
      }
    }

    return pathWithoutReadTheDocsPrefix;
  }

  function getRawRepoBase() {
    const configured = getMetaContent("mkdocs-copy-to-llm-repo-url");
    if (configured) return configured.replace(/\/+$/, "");
    return "https://raw.githubusercontent.com/Longitudinal-Depression-Toolkit/ldt-toolkit/refs/heads/main/docs";
  }

  function buildMarkdownCandidates(path) {
    const stripped = stripBasePath(path);
    const withoutSlashes = stripped.replace(/^\/+|\/+$/g, "");
    const repoBase = getRawRepoBase();
    if (!withoutSlashes) {
      return [`${repoBase}/index.md`];
    }
    return [
      `${repoBase}/${withoutSlashes}.md`,
      `${repoBase}/${withoutSlashes}/index.md`,
    ];
  }

  async function firstReachableUrl(urls) {
    for (const url of urls) {
      try {
        const response = await fetch(url, { method: "HEAD", cache: "no-store" });
        if (response.ok) return url;
      } catch (_) {
        // Try next candidate.
      }
    }
    return urls[0] || "";
  }

  async function resolveMarkdownUrl() {
    if (resolvedMdUrlCache) return resolvedMdUrlCache;
    const path = normalizePath(window.location.pathname);
    const candidates = buildMarkdownCandidates(path);
    resolvedMdUrlCache = await firstReachableUrl(candidates);
    return resolvedMdUrlCache;
  }

  function closeDropdownForItem(item) {
    const container = item.closest(".copy-to-llm-split-container");
    if (!container) return;
    const dropdown = container.querySelector(".copy-to-llm-dropdown");
    const button = container.querySelector(".copy-to-llm-right");
    if (dropdown) dropdown.classList.remove("show");
    if (button) {
      button.classList.remove("active");
      button.setAttribute("aria-expanded", "false");
      const chevron = button.querySelector(".chevron-icon");
      if (chevron) chevron.style.transform = "";
    }
  }

  async function copyText(text) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
  }

  function buildPromptUrl(action, mdUrl) {
    if (action === "open-chatgpt") {
      const prompt = `Read ${mdUrl} so I can ask questions about it.`;
      return `https://chatgpt.com/?hints=search&q=${encodeURIComponent(prompt)}`;
    }
    if (action === "open-claude") {
      const prompt = `Read ${mdUrl} so I can ask questions about it.`;
      return `https://claude.ai/new?q=${encodeURIComponent(prompt)}`;
    }
    return "";
  }

  function handleDropdownAction(event) {
    const item = event.target.closest(".copy-to-llm-dropdown-item");
    if (!item) return;
    const action = item.dataset.action;
    if (!MARKDOWN_ACTIONS.has(action)) return;

    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();

    resolveMarkdownUrl().then(async (mdUrl) => {
      if (!mdUrl) return;

      if (action === "view-markdown") {
        window.open(mdUrl, "_blank");
      } else if (action === "copy-markdown-link") {
        await copyText(mdUrl);
      } else {
        const promptUrl = buildPromptUrl(action, mdUrl);
        if (promptUrl) window.open(promptUrl, "_blank");
      }

      closeDropdownForItem(item);
    });
  }

  function isHomePage(path) {
    if (path === "/") return true;
    return Boolean(document.querySelector(".tx-container"));
  }

  function isExcludedPath(path) {
    if (isHomePage(path)) return true;
    for (const route of EXCLUDED_ROUTE_SUFFIXES) {
      if (path === route || path.endsWith(route)) {
        return true;
      }
    }
    return false;
  }

  function removeCopyToLLMUI() {
    document
      .querySelectorAll(
        ".copy-to-llm, .copy-to-llm-split-container, .copy-to-llm-dropdown, .copy-to-llm-toast"
      )
      .forEach((node) => node.remove());

    document.querySelectorAll(".h1-copy-wrapper").forEach((wrapper) => {
      const h1 = wrapper.querySelector("h1");
      const parent = wrapper.parentNode;
      if (h1 && parent) {
        parent.insertBefore(h1, wrapper);
      }
      wrapper.remove();
    });
  }

  function applyCopyToLLMScope() {
    const path = normalizePath(window.location.pathname);
    if (!isExcludedPath(path)) return;
    removeCopyToLLMUI();
  }

  function scheduleApply() {
    window.setTimeout(applyCopyToLLMScope, 0);
    window.requestAnimationFrame(() => {
      window.setTimeout(applyCopyToLLMScope, 0);
    });
  }

  document.addEventListener("click", handleDropdownAction, true);

  if (typeof window.document$ !== "undefined" && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => {
      resolvedMdUrlCache = null;
      scheduleApply();
    });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleApply);
  } else {
    scheduleApply();
  }
})();
