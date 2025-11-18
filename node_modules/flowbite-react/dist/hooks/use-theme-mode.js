'use client';
import { useState, useEffect } from 'react';
import { isClient } from '../helpers/is-client.js';
import { useWatchLocalStorageValue } from './use-watch-localstorage-value.js';
import { getPrefix, getVersion, getMode } from '../store/index.js';

const DEFAULT_MODE = "auto";
const LS_THEME_MODE = "flowbite-theme-mode";
const SYNC_THEME_MODE = "flowbite-theme-mode-sync";
function useThemeMode() {
  const [mode, setMode] = useState(getInitialMode(getMode()));
  useWatchLocalStorageValue({
    key: LS_THEME_MODE,
    onChange(newMode) {
      setMode(validateMode(newMode ?? DEFAULT_MODE));
    }
  });
  useSyncMode((mode2) => setMode(mode2));
  function handleSetMode(mode2) {
    setMode(mode2);
    setModeInLS(mode2);
    setModeInDOM(mode2);
    document.dispatchEvent(new CustomEvent(SYNC_THEME_MODE, { detail: mode2 }));
  }
  function toggleMode() {
    let newMode = mode;
    if (newMode === "auto") {
      newMode = computeModeValue(newMode);
    }
    newMode = newMode === "dark" ? "light" : "dark";
    handleSetMode(newMode);
  }
  function clearMode() {
    const newMode = mode ?? DEFAULT_MODE;
    handleSetMode(newMode);
  }
  return {
    mode,
    computedMode: computeModeValue(mode),
    setMode: handleSetMode,
    toggleMode,
    clearMode
  };
}
function useSyncMode(onChange) {
  useEffect(() => {
    function handleSync(e) {
      const mode = e.detail;
      onChange(mode);
    }
    document.addEventListener(SYNC_THEME_MODE, handleSync);
    return () => document.removeEventListener(SYNC_THEME_MODE, handleSync);
  }, []);
}
function setModeInLS(mode) {
  localStorage.setItem(LS_THEME_MODE, mode);
}
function setModeInDOM(mode) {
  const computedMode = computeModeValue(mode);
  const prefix = getPrefix() ?? "";
  const version = getVersion();
  const className = version === 3 ? `${prefix}dark` : "dark";
  if (computedMode === "dark") {
    document.documentElement.classList.add(className);
  } else {
    document.documentElement.classList.remove(className);
  }
}
function getInitialMode(defaultMode) {
  if (!isClient()) {
    return DEFAULT_MODE;
  }
  const storageMode = localStorage.getItem(LS_THEME_MODE);
  return validateMode(storageMode ?? defaultMode ?? DEFAULT_MODE);
}
function computeModeValue(mode) {
  if (!isClient()) {
    return DEFAULT_MODE;
  }
  return mode === "auto" ? prefersColorScheme() : mode;
}
function prefersColorScheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function validateMode(mode) {
  if (["light", "dark", "auto"].includes(mode)) {
    return mode;
  }
  return DEFAULT_MODE;
}

export { useThemeMode };
//# sourceMappingURL=use-theme-mode.js.map
