'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { kbdTheme } from './theme.js';

const Kbd = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [kbdTheme, provider.theme?.kbd, props.theme],
    [get(provider.clearTheme, "kbd"), props.clearTheme],
    [get(provider.applyTheme, "kbd"), props.applyTheme]
  );
  const { children, className, icon: Icon, ...restProps } = resolveProps(props, provider.props?.hr);
  return /* @__PURE__ */ jsxs("span", { ref, className: twMerge(theme.root.base, className), "data-testid": "flowbite-kbd", ...restProps, children: [
    Icon && /* @__PURE__ */ jsx(Icon, { className: theme.root.icon, "data-testid": "flowbite-kbd-icon" }),
    children
  ] });
});
Kbd.displayName = "Kbd";

export { Kbd };
//# sourceMappingURL=Kbd.js.map
