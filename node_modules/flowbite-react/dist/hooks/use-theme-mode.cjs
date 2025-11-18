'use client';
'use strict';

var React = require('react');
var isClient = require('../helpers/is-client.cjs');
var useWatchLocalstorageValue = require('./use-watch-localstorage-value.cjs');
var index = require('../store/index.cjs');

const DEFAULT_MODE = "auto";
const LS_THEME_MODE = "flowbite-theme-mode";
const SYNC_THEME_MODE = "flowbite-theme-mode-sync";
function useThemeMode() {
  const [mode, setMode] = React.useState(getInitialMode(index.getMode()));
  useWatchLocalstorageValue.useWatchLocalStorageValue({
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
  React.useEffect(() => {
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
  const prefix = index.getPrefix() ?? "";
  const version = index.getVersion();
  const className = version === 3 ? `${prefix}dark` : "dark";
  if (computedMode === "dark") {
    document.documentElement.classList.add(className);
  } else {
    document.documentElement.classList.remove(className);
  }
}
function getInitialMode(defaultMode) {
  if (!isClient.isClient()) {
    return DEFAULT_MODE;
  }
  const storageMode = localStorage.getItem(LS_THEME_MODE);
  return validateMode(storageMode ?? defaultMode ?? DEFAULT_MODE);
}
function computeModeValue(mode) {
  if (!isClient.isClient()) {
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

exports.useThemeMode = useThemeMode;
//# sourceMappingURL=use-theme-mode.cjs.map
