(() => {
  const HOME_SELECTOR = "[data-homepage-root]";
  const BODY_CLASS = "homepage-page";
  const EXPAND_DELAY_MS = 5000;
  const WAVE_RANGE = 760;
  const MAX_WAVE_SCALE = 1.45;
  const LERP_FACTOR = 0.085;

  let cleanupActivePage = () => {
    resetBodyState();
  };

  function resetBodyState() {
    if (!document.body) return;
    document.body.classList.remove(BODY_CLASS);
    document.body.style.removeProperty("--footer-wave-progress");
    document.body.style.removeProperty("--footer-wave-offset-y");
  }

  function cleanupHomepage() {
    cleanupActivePage();
    cleanupActivePage = () => {
      resetBodyState();
    };
  }

  function trackedTimeouts() {
    const ids = new Set();

    return {
      add(callback, delayMs) {
        const id = window.setTimeout(() => {
          ids.delete(id);
          callback();
        }, delayMs);
        ids.add(id);
        return id;
      },
      clear(id) {
        if (id === null || typeof id === "undefined") return;
        ids.delete(id);
        window.clearTimeout(id);
      },
      clearAll() {
        ids.forEach((id) => window.clearTimeout(id));
        ids.clear();
      },
    };
  }

  function trackedFrames() {
    const ids = new Set();

    return {
      add(callback) {
        const id = window.requestAnimationFrame((time) => {
          ids.delete(id);
          callback(time);
        });
        ids.add(id);
        return id;
      },
      clearAll() {
        ids.forEach((id) => window.cancelAnimationFrame(id));
        ids.clear();
      },
    };
  }

  function initHomepage(root = document) {
    cleanupHomepage();

    const homeRoot = root.querySelector(HOME_SELECTOR);
    if (!homeRoot || !document.body) {
      return;
    }

    const docEl = document.documentElement;
    const body = document.body;
    const ua = navigator.userAgent || "";
    const isSafari =
      /safari/i.test(ua) &&
      !/chrome|chromium|android|crios|fxios|edgios/i.test(ua);

    if (isSafari) {
      docEl.classList.add("is-safari");
    }

    body.classList.add(BODY_CLASS);
    body.style.setProperty("--footer-wave-progress", "0");
    body.style.setProperty("--footer-wave-offset-y", "0px");

    const controller = new AbortController();
    const { signal } = controller;
    const timeouts = trackedTimeouts();
    const frames = trackedFrames();
    const cleanupTasks = [];

    const heroShell = homeRoot.querySelector(".tx-landing__media-shell");
    const heroPlayer = homeRoot.querySelector(".tx-landing__media");
    let expandTimer = null;

    if (heroShell) {
      const positionExpandedHero = () => {
        const wrap = heroShell.querySelector(".tx-landing__media-wrap");
        if (!wrap) return;

        const rect = wrap.getBoundingClientRect();
        const currentCenterX = rect.left + rect.width / 2;
        const currentCenterY = rect.top + rect.height / 2;
        const targetCenterX = window.innerWidth / 2;
        const targetCenterY = window.innerHeight * 0.62;
        const dx = targetCenterX - currentCenterX;
        const dy = targetCenterY - currentCenterY;

        heroShell.style.setProperty("--hero-expand-x", `${dx.toFixed(1)}px`);
        heroShell.style.setProperty("--hero-expand-y", `${dy.toFixed(1)}px`);
      };

      const clearExpandTimer = () => {
        timeouts.clear(expandTimer);
        expandTimer = null;
      };

      const collapseHeroPreview = () => {
        clearExpandTimer();
        heroShell.classList.remove("is-expanded");
      };

      const clearPointerFocusFromHero = () => {
        const activeElement = document.activeElement;
        if (!activeElement) return;
        if (activeElement === heroPlayer || heroShell.contains(activeElement)) {
          if (typeof activeElement.blur === "function") {
            activeElement.blur();
          }
        }
      };

      const scheduleHeroExpansion = () => {
        clearExpandTimer();
        expandTimer = timeouts.add(() => {
          positionExpandedHero();
          heroShell.classList.add("is-expanded");
          expandTimer = null;
        }, EXPAND_DELAY_MS);
      };

      const onHeroResize = () => {
        if (heroShell.classList.contains("is-expanded")) {
          positionExpandedHero();
        }
      };

      heroShell.addEventListener("pointerenter", scheduleHeroExpansion, { signal });
      heroShell.addEventListener(
        "pointerleave",
        () => {
          collapseHeroPreview();
          clearPointerFocusFromHero();
        },
        { signal }
      );
      heroShell.addEventListener("focusin", scheduleHeroExpansion, { signal });
      heroShell.addEventListener("focusout", collapseHeroPreview, { signal });
      window.addEventListener("resize", onHeroResize, { signal });

      cleanupTasks.push(() => {
        clearExpandTimer();
        heroShell.classList.remove("is-expanded");
      });
    }

    if (heroPlayer) {
      heroPlayer.muted = false;
      heroPlayer.defaultMuted = false;
      heroPlayer.volume = 1;

      if (typeof heroPlayer.play === "function") {
        const autoPlayPromise = heroPlayer.play();
        if (autoPlayPromise && typeof autoPlayPromise.catch === "function") {
          autoPlayPromise.catch(() => {
            heroPlayer.muted = true;
            heroPlayer.defaultMuted = true;

            const mutedFallbackPromise = heroPlayer.play();
            if (
              mutedFallbackPromise &&
              typeof mutedFallbackPromise.catch === "function"
            ) {
              mutedFallbackPromise.catch(() => {});
            }
          });
        }
      }

      const enableHeroAudio = () => {
        heroPlayer.muted = false;
        heroPlayer.defaultMuted = false;
        heroPlayer.volume = 1;

        if (typeof heroPlayer.play === "function") {
          const playPromise = heroPlayer.play();
          if (playPromise && typeof playPromise.catch === "function") {
            playPromise.catch(() => {});
          }
        }
      };

      window.addEventListener("pointerdown", enableHeroAudio, {
        once: true,
        passive: true,
        signal,
      });
      window.addEventListener("keydown", enableHeroAudio, {
        once: true,
        signal,
      });
    }

    const items = Array.from(homeRoot.querySelectorAll(".podcast-item"));
    const bubbleTray = homeRoot.querySelector(".podcast-bubbles");

    const resetOthers = (activeItem) => {
      items.forEach((item) => {
        const audioEl = item.querySelector(".podcast-audio");
        const isActive = item === activeItem;
        item.classList.toggle("active", isActive);
        if (!isActive && audioEl) {
          audioEl.pause();
          audioEl.currentTime = 0;
        }
      });
    };

    items.forEach((item, index) => {
      const bubble = item.querySelector(".podcast-bubble");
      const audioEl = item.querySelector(".podcast-audio");
      if (!bubble) return;

      if (!bubble.style.getPropertyValue("--delay")) {
        bubble.style.setProperty("--delay", (index * 0.12).toFixed(2));
      }

      if (audioEl) {
        audioEl.playbackRate = 1.25;
        audioEl.addEventListener("play", () => audioEl.classList.add("playing"), {
          signal,
        });
        audioEl.addEventListener(
          "pause",
          () => audioEl.classList.remove("playing"),
          { signal }
        );
      }

      bubble.addEventListener(
        "click",
        () => {
          resetOthers(item);
          if (audioEl) {
            audioEl.pause();
            audioEl.currentTime = 0;
          }
        },
        { signal }
      );
    });

    if (items.length) {
      resetOthers(items[0]);
    }

    if (bubbleTray) {
      timeouts.add(() => bubbleTray.classList.remove("intro"), 1200);
    }

    cleanupTasks.push(() => {
      items.forEach((item) => {
        const audioEl = item.querySelector(".podcast-audio");
        if (!audioEl) return;
        audioEl.pause();
        audioEl.currentTime = 0;
      });
    });

    const footer = homeRoot.querySelector(".footer");
    const footerBeyond = homeRoot.querySelector("[data-footer-beyond]");

    if (footer && footerBeyond) {
      let currentScale = 0;
      let targetScale = 0;
      let footerFramePending = false;

      if (isSafari) {
        footerBeyond.querySelectorAll("g[filter]").forEach((group) => {
          group.removeAttribute("filter");
        });
      }

      const clamp01 = (value) => Math.min(1, Math.max(0, value));

      const getTargetScale = () => {
        const viewportBottom = window.scrollY + window.innerHeight;
        const nearFooterDistance = viewportBottom - footer.offsetTop;
        const nearFooterProgress = clamp01(nearFooterDistance / WAVE_RANGE);
        const nativeOverscroll = Math.max(0, viewportBottom - docEl.scrollHeight);
        const overscrollProgress = clamp01(nativeOverscroll / WAVE_RANGE);
        return Math.max(nearFooterProgress, overscrollProgress) * MAX_WAVE_SCALE;
      };

      const applyWaveScale = (scale) => {
        const overlayHeight =
          footerBeyond.getBoundingClientRect().height || window.innerHeight;
        const waveOffsetPx = scale * overlayHeight;

        footerBeyond.style.transform = `scaleY(${scale.toFixed(6)})`;
        body.style.setProperty("--footer-wave-progress", scale.toFixed(6));
        body.style.setProperty(
          "--footer-wave-offset-y",
          `${waveOffsetPx.toFixed(2)}px`
        );
      };

      const animateFooterBeyond = () => {
        footerFramePending = false;
        const delta = targetScale - currentScale;
        currentScale += delta * LERP_FACTOR;

        if (Math.abs(delta) < 0.0005) {
          currentScale = targetScale;
        }

        applyWaveScale(currentScale);

        if (currentScale !== targetScale) {
          footerFramePending = true;
          frames.add(animateFooterBeyond);
        }
      };

      const scheduleRender = () => {
        targetScale = getTargetScale();
        if (!footerFramePending) {
          footerFramePending = true;
          frames.add(animateFooterBeyond);
        }
      };

      window.addEventListener("scroll", scheduleRender, {
        passive: true,
        signal,
      });
      window.addEventListener("wheel", scheduleRender, {
        passive: true,
        signal,
      });
      window.addEventListener("touchmove", scheduleRender, {
        passive: true,
        signal,
      });
      window.addEventListener("resize", scheduleRender, { signal });
      window.addEventListener("orientationchange", scheduleRender, { signal });
      scheduleRender();

      cleanupTasks.push(() => {
        footerBeyond.style.transform = "scaleY(0)";
      });
    }

    cleanupActivePage = () => {
      controller.abort();
      timeouts.clearAll();
      frames.clearAll();
      cleanupTasks.forEach((task) => task());
      resetBodyState();
    };
  }

  function scheduleInit() {
    window.requestAnimationFrame(() => initHomepage(document));
  }

  if (
    typeof window.document$ !== "undefined" &&
    typeof window.document$.subscribe === "function"
  ) {
    window.document$.subscribe(() => {
      scheduleInit();
    });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleInit);
  } else {
    scheduleInit();
  }
})();
