'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var SidebarContext = require('./SidebarContext.cjs');
var theme = require('./theme.cjs');

const Sidebar = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme, provider$1.theme?.sidebar, props.theme],
    [get.get(provider$1.clearTheme, "sidebar"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar"), props.applyTheme]
  );
  const {
    as: Component = "nav",
    children,
    className,
    collapseBehavior = "collapse",
    collapsed: isCollapsed = false,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.sidebar);
  return /* @__PURE__ */ jsxRuntime.jsx(
    SidebarContext.SidebarContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, isCollapsed },
      children: /* @__PURE__ */ jsxRuntime.jsx(
        Component,
        {
          ref,
          "aria-label": "Sidebar",
          hidden: isCollapsed && collapseBehavior === "hide",
          className: tailwindMerge.twMerge(theme$1.root.base, theme$1.root.collapsed[isCollapsed ? "on" : "off"], className),
          ...restProps,
          children: /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.root.inner, children })
        }
      )
    }
  );
});
Sidebar.displayName = "Sidebar";

exports.Sidebar = Sidebar;
//# sourceMappingURL=Sidebar.cjs.map
