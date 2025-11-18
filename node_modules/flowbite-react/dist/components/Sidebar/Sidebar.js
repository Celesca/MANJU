'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { SidebarContext } from './SidebarContext.js';
import { sidebarTheme } from './theme.js';

const Sidebar = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme, provider.theme?.sidebar, props.theme],
    [get(provider.clearTheme, "sidebar"), props.clearTheme],
    [get(provider.applyTheme, "sidebar"), props.applyTheme]
  );
  const {
    as: Component = "nav",
    children,
    className,
    collapseBehavior = "collapse",
    collapsed: isCollapsed = false,
    ...restProps
  } = resolveProps(props, provider.props?.sidebar);
  return /* @__PURE__ */ jsx(
    SidebarContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, isCollapsed },
      children: /* @__PURE__ */ jsx(
        Component,
        {
          ref,
          "aria-label": "Sidebar",
          hidden: isCollapsed && collapseBehavior === "hide",
          className: twMerge(theme.root.base, theme.root.collapsed[isCollapsed ? "on" : "off"], className),
          ...restProps,
          children: /* @__PURE__ */ jsx("div", { className: theme.root.inner, children })
        }
      )
    }
  );
});
Sidebar.displayName = "Sidebar";

export { Sidebar };
//# sourceMappingURL=Sidebar.js.map
