function copyTextFromSelector(selector) {
  const el = document.querySelector(selector);
  if (!el) return false;

  const text = el.innerText;
  if (!text) return false;

  const copyWithClipboardApi = async () => {
    await navigator.clipboard.writeText(text);
    return true;
  };

  const copyWithFallback = () => {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  };

  if (navigator.clipboard && window.isSecureContext) {
    return copyWithClipboardApi().then(() => true).catch(() => copyWithFallback());
  }
  return Promise.resolve(copyWithFallback());
}

function wireCopyButtons() {
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const selector = btn.getAttribute("data-copy");
      const ok = await copyTextFromSelector(selector);
      const old = btn.innerText;
      btn.innerText = ok ? "Copied" : "Failed";
      btn.disabled = true;
      setTimeout(() => {
        btn.innerText = old;
        btn.disabled = false;
      }, 900);
    });
  });
}

wireCopyButtons();

function once(el, eventName, handler, opts) {
  const wrapped = (ev) => {
    el.removeEventListener(eventName, wrapped, opts);
    handler(ev);
  };
  el.addEventListener(eventName, wrapped, opts);
}

function waitForEvent(el, eventName, timeoutMs) {
  return new Promise((resolve, reject) => {
    const onEvent = () => {
      cleanup();
      resolve();
    };
    const onTimeout = () => {
      cleanup();
      reject(new Error(`timeout waiting for ${eventName}`));
    };
    const cleanup = () => {
      el.removeEventListener(eventName, onEvent);
      if (timer) clearTimeout(timer);
    };
    el.addEventListener(eventName, onEvent, { passive: true });
    const timer = timeoutMs ? setTimeout(onTimeout, timeoutMs) : null;
  });
}

async function ensureVideoReadyForCapture(video) {
  if (!video) return false;
  if (video.getAttribute("poster")) return true;

  if (!video.getAttribute("preload") || video.getAttribute("preload") === "metadata") {
    video.setAttribute("preload", "auto");
  }
  if (!video.muted) video.muted = true;
  video.playsInline = true;
  try {
    const src = video.currentSrc || video.src || "";
    const url = new URL(src, window.location.href);
    if (url.origin !== window.location.origin) video.crossOrigin = video.crossOrigin || "anonymous";
  } catch {
    // ignore
  }

  try {
    video.load();
  } catch {
    // ignore
  }

  // Some browsers (esp. mobile) may delay loading until user gesture; don't hang forever.
  if (video.readyState < 1) {
    try {
      await waitForEvent(video, "loadedmetadata", 2500);
    } catch {
      return false;
    }
  }

  // Ensure dimensions and a decodable frame exist
  if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
    try {
      await waitForEvent(video, "loadeddata", 2500);
    } catch {
      // Still allow capture attempts; may succeed after user presses play.
    }
  }

  return true;
}

function getVideoSrc(video) {
  if (!video) return "";
  if (video.currentSrc) return video.currentSrc;
  if (video.src) return video.src;
  const source = video.querySelector("source[src]");
  return source ? source.getAttribute("src") || "" : "";
}

async function capturePosterFromSecondFrame(video) {
  if (!video) return false;
  if (video.getAttribute("poster")) return false;
  const src = getVideoSrc(video);
  if (!src) return false;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return false;

  const looksBlank = () => {
    try {
      const sw = Math.min(16, canvas.width);
      const sh = Math.min(16, canvas.height);
      const img = ctx.getImageData(0, 0, sw, sh).data;
      let sum = 0;
      for (let i = 0; i < img.length; i += 4) {
        const r = img[i];
        const g = img[i + 1];
        const b = img[i + 2];
        // Perceived luminance (rough)
        sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
      }
      const avg = sum / (img.length / 4);
      return avg < 8;
    } catch {
      return false;
    }
  };

  const captureFrom = async (v, t) => {
    try {
      v.currentTime = t;
    } catch {
      // ignore
    }
    try {
      await waitForEvent(v, "seeked", 5000);
    } catch {
      // ignore seek timeout
    }
    if (!v.videoWidth || !v.videoHeight) return false;
    if (v.readyState < 2) {
      try {
        await waitForEvent(v, "loadeddata", 5000);
      } catch {
        // ignore
      }
    }
    try {
      canvas.width = v.videoWidth;
      canvas.height = v.videoHeight;
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
      if (looksBlank()) return false;
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.86));
      if (blob) {
        const objectUrl = URL.createObjectURL(blob);
        const old = video.dataset.autoPosterUrl;
        if (old) URL.revokeObjectURL(old);
        video.dataset.autoPosterUrl = objectUrl;
        video.setAttribute("poster", objectUrl);
        return true;
      }
      const dataUrl = canvas.toDataURL("image/jpeg", 0.86);
      if (dataUrl && dataUrl.startsWith("data:image/")) {
        video.setAttribute("poster", dataUrl);
        return true;
      }
    } catch {
      // ignore and fallback
    }
    return false;
  };

  const probe = document.createElement("video");
  probe.muted = true;
  probe.playsInline = true;
  probe.preload = "auto";
  try {
    const url = new URL(src, window.location.href);
    if (url.origin !== window.location.origin) probe.crossOrigin = "anonymous";
  } catch {
    // ignore
  }
  probe.src = src;
  try {
    probe.load();
  } catch {
    // ignore
  }

  let duration = 0;
  try {
    await waitForEvent(probe, "loadedmetadata", 5000);
    duration = Number.isFinite(probe.duration) ? probe.duration : 0;
  } catch {
    // If we can't load without a user gesture, defer and try again later.
    once(video, "play", () => capturePosterFromSecondFrame(video).catch(() => {}), { passive: true });
    once(video, "loadeddata", () => capturePosterFromSecondFrame(video).catch(() => {}), { passive: true });
    return false;
  }

  const attempts = [1 / 30, 1 / 24, 0.1, 0.15, 0.3, 0.5].map((t) =>
    duration ? Math.min(Math.max(t, 0.001), Math.max(duration - 0.001, 0.001)) : t,
  );

  let ok = false;
  for (const t of attempts) {
    ok = await captureFrom(probe, t);
    if (ok) break;
  }

  // Fallback: try capturing from the on-page video element.
  if (!ok) {
    const ready = await ensureVideoReadyForCapture(video);
    if (ready) {
      for (const t of attempts) {
        ok = await captureFrom(video, t);
        if (ok) break;
      }
    }
  }

  try {
    probe.removeAttribute("src");
    probe.load();
  } catch {
    // ignore
  }

  return ok;
}

function wireAutoPosters() {
  const videos = Array.from(document.querySelectorAll("video")).filter((v) => !v.getAttribute("poster"));
  if (!videos.length) return;

  const schedule = (video) => {
    capturePosterFromSecondFrame(video).catch(() => {});
  };

  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          io.unobserve(e.target);
          schedule(e.target);
        });
      },
      { rootMargin: "200px 0px", threshold: 0.01 },
    );
    videos.forEach((v) => io.observe(v));
    return;
  }

  videos.forEach(schedule);
}

wireAutoPosters();

function wireAutoLoopPlay() {
  const videos = Array.from(document.querySelectorAll("video"));
  if (!videos.length) return;

  videos.forEach((video) => {
    video.loop = true;
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    if (!video.getAttribute("preload") || video.getAttribute("preload") === "metadata") {
      video.setAttribute("preload", "auto");
    }
  });

  const tryPlayAll = () => {
    videos.forEach((video) => {
      try {
        const p = video.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } catch {
        // ignore
      }
    });
  };

  if (document.readyState === "complete" || document.readyState === "interactive") {
    tryPlayAll();
  } else {
    window.addEventListener("DOMContentLoaded", tryPlayAll, { once: true, passive: true });
  }
  window.addEventListener("pageshow", tryPlayAll, { passive: true });
}

wireAutoLoopPlay();
