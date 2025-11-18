'use client';
import { jsx } from 'react/jsx-runtime';
import { deepmerge } from 'deepmerge-ts';
import { createContext, useContext, useMemo } from 'react';
import { deepMergeStrings } from '../helpers/deep-merge.js';
import { twMerge } from '../helpers/tailwind-merge.js';

const ThemeProviderContext = createContext(void 0);
function ThemeProvider({ children, root, props, theme, clearTheme, applyTheme }) {
  const parentProvider = useContext(ThemeProviderContext);
  const value = useMemo(
    () => ({
      props: !root && parentProvider?.props ? deepmerge(parentProvider?.props, props) : props,
      theme: !root && parentProvider?.theme ? deepMergeStrings(twMerge)(parentProvider.theme, theme) : theme,
      clearTheme: !root && parentProvider?.clearTheme ? deepmerge(parentProvider.clearTheme, clearTheme) : clearTheme,
      applyTheme: !root && parentProvider?.applyTheme ? deepmerge(parentProvider?.applyTheme, applyTheme) : applyTheme
    }),
    [
      root,
      props,
      theme,
      clearTheme,
      applyTheme,
      parentProvider?.props,
      parentProvider?.theme,
      parentProvider?.clearTheme,
      parentProvider?.applyTheme
    ]
  );
  return /* @__PURE__ */ jsx(ThemeProviderContext.Provider, { value, children });
}
ThemeProvider.displayName = "ThemeProvider";
function useThemeProvider() {
  return useContext(ThemeProviderContext) ?? {};
}

export { ThemeProvider, useThemeProvider };
//# sourceMappingURL=provider.js.map
