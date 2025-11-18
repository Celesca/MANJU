'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { badgeTheme } from './theme.js';

const Badge = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [badgeTheme, provider.theme?.badge, props.theme],
    [get(provider.clearTheme, "badge"), props.clearTheme],
    [get(provider.applyTheme, "badge"), props.applyTheme]
  );
  const {
    children,
    color = "info",
    icon: Icon,
    size = "xs",
    className,
    ...restProps
  } = resolveProps(props, provider.props?.badge);
  return /* @__PURE__ */ jsxs(
    "span",
    {
      ref,
      className: twMerge(
        theme.root.base,
        theme.root.color[color],
        theme.root.size[size],
        theme.icon[Icon ? "on" : "off"],
        className
      ),
      "data-testid": "flowbite-badge",
      ...restProps,
      children: [
        Icon && /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, className: theme.icon.size[size], "data-testid": "flowbite-badge-icon" }),
        children && /* @__PURE__ */ jsx("span", { children })
      ]
    }
  );
});
Badge.displayName = "Badge";

export { Badge };
//# sourceMappingURL=Badge.js.map
