import { jsx } from 'react/jsx-runtime';

const defaultOptions = {
  defaultMode: "auto",
  localStorageKey: "flowbite-theme-mode",
  prefix: "",
  version: 4
};
function ThemeModeScript({
  mode,
  defaultMode = defaultOptions.defaultMode,
  localStorageKey = defaultOptions.localStorageKey,
  prefix = defaultOptions.prefix,
  version = defaultOptions.version,
  ...others
}) {
  return /* @__PURE__ */ jsx(
    "script",
    {
      ...others,
      "data-flowbite-theme-mode-script": true,
      dangerouslySetInnerHTML: {
        __html: getThemeModeScript({
          mode,
          defaultMode,
          localStorageKey,
          prefix,
          version
        })
      }
    }
  );
}
ThemeModeScript.displayName = "ThemeModeScript";
function getThemeModeScript(props = {}) {
  const {
    mode,
    defaultMode = defaultOptions.defaultMode,
    localStorageKey = defaultOptions.localStorageKey,
    prefix = defaultOptions.prefix,
    version = defaultOptions.version
  } = props;
  return `
    try {
      const storageMode = window.localStorage.getItem("${localStorageKey}");
      const isStorageModeValid = storageMode === "light" || storageMode === "dark" || storageMode === "auto";
      const resolvedMode = (isStorageModeValid ? storageMode : null) ?? ${mode ? `"${mode}"` : void 0} ?? "${defaultMode}";
      const computedMode =
        resolvedMode === "auto" ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : resolvedMode;
      const className = ${version === 3 ? `"${prefix}dark"` : `"dark"`};

      if (computedMode === "dark") {
        document.documentElement.classList.add(className);
      } else {
        document.documentElement.classList.remove(className);
      }
      localStorage.setItem("${localStorageKey}", resolvedMode);

      // Add listener for system theme changes
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      mediaQuery.addEventListener("change", (e) => {
        const storageMode = window.localStorage.getItem("${localStorageKey}");
        const isStorageModeValid = storageMode === "light" || storageMode === "dark" || storageMode === "auto";
        const resolvedMode = isStorageModeValid ? storageMode : "${defaultMode}";

        if (resolvedMode === "auto") {
          if (e.matches) {
            document.documentElement.classList.add(className);
          } else {
            document.documentElement.classList.remove(className);
          }
        }
      });

      // Add listener for storage changes
      window.addEventListener("storage", (e) => {
        if (e.key === "${localStorageKey}") {
          const newMode = e.newValue;
          const isStorageModeValid = newMode === "light" || newMode === "dark" || newMode === "auto";
          const resolvedMode = isStorageModeValid ? newMode : "${defaultMode}";

          if (resolvedMode === "dark" || (resolvedMode === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
            document.documentElement.classList.add(className);
          } else {
            document.documentElement.classList.remove(className);
          }
        }
      });
    } catch (e) {}
  `;
}
function initThemeMode(props = {}) {
  const {
    mode,
    defaultMode = defaultOptions.defaultMode,
    localStorageKey = defaultOptions.localStorageKey,
    prefix = defaultOptions.prefix,
    version = defaultOptions.version
  } = props;
  try {
    const storageMode = window.localStorage.getItem(localStorageKey);
    const isStorageModeValid = storageMode === "light" || storageMode === "dark" || storageMode === "auto";
    const resolvedMode = (isStorageModeValid ? storageMode : null) ?? mode ?? defaultMode;
    const computedMode = resolvedMode === "auto" ? window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light" : resolvedMode;
    const className = version === 3 ? `${prefix}dark` : "dark";
    if (computedMode === "dark") {
      document.documentElement.classList.add(className);
    } else {
      document.documentElement.classList.remove(className);
    }
    localStorage.setItem(localStorageKey, resolvedMode);
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener("change", (e) => {
      const storageMode2 = window.localStorage.getItem(localStorageKey);
      const isStorageModeValid2 = storageMode2 === "light" || storageMode2 === "dark" || storageMode2 === "auto";
      const resolvedMode2 = isStorageModeValid2 ? storageMode2 : defaultMode;
      if (resolvedMode2 === "auto") {
        if (e.matches) {
          document.documentElement.classList.add(className);
        } else {
          document.documentElement.classList.remove(className);
        }
      }
    });
    window.addEventListener("storage", (e) => {
      if (e.key === localStorageKey) {
        const newMode = e.newValue;
        const isStorageModeValid2 = newMode === "light" || newMode === "dark" || newMode === "auto";
        const resolvedMode2 = isStorageModeValid2 ? newMode : defaultMode;
        if (resolvedMode2 === "dark" || resolvedMode2 === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches) {
          document.documentElement.classList.add(className);
        } else {
          document.documentElement.classList.remove(className);
        }
      }
    });
  } catch (e) {
  }
}

export { ThemeModeScript, getThemeModeScript, initThemeMode };
//# sourceMappingURL=mode-script.js.map
