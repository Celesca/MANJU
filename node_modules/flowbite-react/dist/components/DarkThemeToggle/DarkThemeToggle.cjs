'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var useThemeMode = require('../../hooks/use-theme-mode.cjs');
var moonIcon = require('../../icons/moon-icon.cjs');
var sunIcon = require('../../icons/sun-icon.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const DarkThemeToggle = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.darkThemeToggleTheme, provider$1.theme?.darkThemeToggle, props.theme],
    [get.get(provider$1.clearTheme, "darkThemeToggle"), props.clearTheme],
    [get.get(provider$1.applyTheme, "darkThemeToggle"), props.applyTheme]
  );
  const {
    className,
    iconDark: IconDark = sunIcon.SunIcon,
    iconLight: IconLight = moonIcon.MoonIcon,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.darkThemeToggle);
  const { toggleMode } = useThemeMode.useThemeMode();
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "button",
    {
      ref,
      type: "button",
      "aria-label": "Toggle dark mode",
      "data-testid": "dark-theme-toggle",
      className: tailwindMerge.twMerge(theme$1.root.base, className),
      onClick: toggleMode,
      ...restProps,
      children: [
        /* @__PURE__ */ jsxRuntime.jsx(IconDark, { "aria-label": "Currently dark mode", className: tailwindMerge.twMerge(theme$1.root.icon.base, theme$1.root.icon.dark) }),
        /* @__PURE__ */ jsxRuntime.jsx(IconLight, { "aria-label": "Currently light mode", className: tailwindMerge.twMerge(theme$1.root.icon.base, theme$1.root.icon.light) })
      ]
    }
  );
});
DarkThemeToggle.displayName = "DarkThemeToggle";

exports.DarkThemeToggle = DarkThemeToggle;
//# sourceMappingURL=DarkThemeToggle.cjs.map
