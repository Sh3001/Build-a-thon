import { useEffect } from "react";

/**
 * Arrow-key navigation for a table. Rows must carry `data-id` and `tabIndex={0}`.
 * Bound to the container rather than the document so two tables never fight.
 */
export function useTableNav(ref, onOpen) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onKey = (e) => {
      const rows = [...el.querySelectorAll("tbody tr[data-id]")];
      if (!rows.length) return;
      const i = rows.indexOf(document.activeElement);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const step = e.key === "ArrowDown" ? 1 : -1;
        rows[Math.max(0, Math.min(rows.length - 1, i < 0 ? 0 : i + step))].focus();
      } else if ((e.key === "Enter" || e.key === " ") && i >= 0) {
        e.preventDefault();
        onOpen(rows[i].dataset.id);
      }
    };
    el.addEventListener("keydown", onKey);
    return () => el.removeEventListener("keydown", onKey);
  }, [ref, onOpen]);
}

/** `/` focuses the search box, `Escape` clears it -- unless you are already typing. */
export function useSearchKeys(ref, onClear) {
  useEffect(() => {
    const onKey = (e) => {
      const t = document.activeElement;
      const typing = t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName);
      if (e.key === "/" && !typing) {
        e.preventDefault();
        ref.current?.focus();
        ref.current?.select();
      } else if (e.key === "Escape" && t === ref.current) {
        onClear();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [ref, onClear]);
}

/** Filter choices that survive a reload. Falls back silently in private windows. */
export function loadPrefs(key, fallback) {
  try {
    return { ...fallback, ...JSON.parse(localStorage.getItem(key) || "{}") };
  } catch {
    return fallback;
  }
}
export function savePrefs(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable -- filters just do not persist */
  }
}
