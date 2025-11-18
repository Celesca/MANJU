'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var deepmergeTs = require('deepmerge-ts');
var React = require('react');
var deepMerge = require('../helpers/deep-merge.cjs');
var tailwindMerge = require('../helpers/tailwind-merge.cjs');

const ThemeProviderContext = React.createContext(void 0);
function ThemeProvider({ children, root, props, theme, clearTheme, applyTheme }) {
  const parentProvider = React.useContext(ThemeProviderContext);
  const value = React.useMemo(
    () => ({
      props: !root && parentProvider?.props ? deepmergeTs.deepmerge(parentProvider?.props, props) : props,
      theme: !root && parentProvider?.theme ? deepMerge.deepMergeStrings(tailwindMerge.twMerge)(parentProvider.theme, theme) : theme,
      clearTheme: !root && parentProvider?.clearTheme ? deepmergeTs.deepmerge(parentProvider.clearTheme, clearTheme) : clearTheme,
      applyTheme: !root && parentProvider?.applyTheme ? deepmergeTs.deepmerge(parentProvider?.applyTheme, applyTheme) : applyTheme
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
  return /* @__PURE__ */ jsxRuntime.jsx(ThemeProviderContext.Provider, { value, children });
}
ThemeProvider.displayName = "ThemeProvider";
function useThemeProvider() {
  return React.useContext(ThemeProviderContext) ?? {};
}

exports.ThemeProvider = ThemeProvider;
exports.useThemeProvider = useThemeProvider;
//# sourceMappingURL=provider.cjs.map
