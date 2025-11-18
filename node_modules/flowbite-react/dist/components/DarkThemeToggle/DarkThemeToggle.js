'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeMode } from '../../hooks/use-theme-mode.js';
import { MoonIcon } from '../../icons/moon-icon.js';
import { SunIcon } from '../../icons/sun-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { darkThemeToggleTheme } from './theme.js';

const DarkThemeToggle = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [darkThemeToggleTheme, provider.theme?.darkThemeToggle, props.theme],
    [get(provider.clearTheme, "darkThemeToggle"), props.clearTheme],
    [get(provider.applyTheme, "darkThemeToggle"), props.applyTheme]
  );
  const {
    className,
    iconDark: IconDark = SunIcon,
    iconLight: IconLight = MoonIcon,
    ...restProps
  } = resolveProps(props, provider.props?.darkThemeToggle);
  const { toggleMode } = useThemeMode();
  return /* @__PURE__ */ jsxs(
    "button",
    {
      ref,
      type: "button",
      "aria-label": "Toggle dark mode",
      "data-testid": "dark-theme-toggle",
      className: twMerge(theme.root.base, className),
      onClick: toggleMode,
      ...restProps,
      children: [
        /* @__PURE__ */ jsx(IconDark, { "aria-label": "Currently dark mode", className: twMerge(theme.root.icon.base, theme.root.icon.dark) }),
        /* @__PURE__ */ jsx(IconLight, { "aria-label": "Currently light mode", className: twMerge(theme.root.icon.base, theme.root.icon.light) })
      ]
    }
  );
});
DarkThemeToggle.displayName = "DarkThemeToggle";

export { DarkThemeToggle };
//# sourceMappingURL=DarkThemeToggle.js.map
