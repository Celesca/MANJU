'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef, useId } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useSidebarContext } from './SidebarContext.js';
import { sidebarTheme } from './theme.js';

const SidebarLogo = forwardRef((props, ref) => {
  const id = useId();
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = useSidebarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme.logo, provider.theme?.sidebar?.logo, rootTheme?.logo, props.theme],
    [get(provider.clearTheme, "sidebar.logo"), get(rootClearTheme, "logo"), props.clearTheme],
    [get(provider.applyTheme, "sidebar.logo"), get(rootApplyTheme, "logo"), props.applyTheme]
  );
  const {
    children,
    className,
    href,
    img,
    imgAlt = "",
    ...restProps
  } = resolveProps(props, provider.props?.sidebarLogo);
  return /* @__PURE__ */ jsxs(
    "a",
    {
      ref,
      "aria-labelledby": `flowbite-sidebar-logo-${id}`,
      href,
      className: twMerge(theme.base, className),
      ...restProps,
      children: [
        /* @__PURE__ */ jsx("img", { alt: imgAlt, src: img, className: theme.img }),
        /* @__PURE__ */ jsx("span", { className: theme.collapsed[isCollapsed ? "on" : "off"], id: `flowbite-sidebar-logo-${id}`, children })
      ]
    }
  );
});
SidebarLogo.displayName = "SidebarLogo";

export { SidebarLogo };
//# sourceMappingURL=SidebarLogo.js.map
