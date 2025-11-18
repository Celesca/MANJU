'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var Badge = require('../Badge/Badge.cjs');
var Tooltip = require('../Tooltip/Tooltip.cjs');
var SidebarContext = require('./SidebarContext.cjs');
var SidebarItemContext = require('./SidebarItemContext.cjs');
var theme = require('./theme.cjs');

const SidebarItem = React.forwardRef((props, ref) => {
  const id = React.useId();
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = SidebarContext.useSidebarContext();
  const { isInsideCollapse } = SidebarItemContext.useSidebarItemContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme.item, provider$1.theme?.sidebar?.item, rootTheme?.item, props.theme],
    [get.get(provider$1.clearTheme, "sidebar.item"), get.get(rootClearTheme, "item"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar.item"), get.get(rootApplyTheme, "item"), props.applyTheme]
  );
  const {
    active: isActive,
    as: Component = "a",
    children,
    className,
    icon: Icon,
    label,
    labelColor = "info",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.sidebarItem);
  return /* @__PURE__ */ jsxRuntime.jsx(ListItem, { theme: theme$1, className: theme$1.listItem, id, isCollapsed, tooltipChildren: children, children: /* @__PURE__ */ jsxRuntime.jsxs(
    Component,
    {
      "aria-labelledby": `flowbite-sidebar-item-${id}`,
      ref,
      className: tailwindMerge.twMerge(
        theme$1.base,
        isActive && theme$1.active,
        !isCollapsed && isInsideCollapse && theme$1.collapsed.insideCollapse,
        className
      ),
      ...restProps,
      children: [
        Icon && /* @__PURE__ */ jsxRuntime.jsx(
          Icon,
          {
            "aria-hidden": true,
            "data-testid": "flowbite-sidebar-item-icon",
            className: tailwindMerge.twMerge(theme$1.icon.base, isActive && theme$1.icon.active)
          }
        ),
        isCollapsed && !Icon && /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.collapsed.noIcon, children: children.charAt(0).toLocaleUpperCase() ?? "?" }),
        !isCollapsed && /* @__PURE__ */ jsxRuntime.jsx(Children, { id, theme: theme$1, children }),
        !isCollapsed && label && /* @__PURE__ */ jsxRuntime.jsx(Badge.Badge, { color: labelColor, "data-testid": "flowbite-sidebar-label", hidden: isCollapsed, className: theme$1.label, children: label })
      ]
    }
  ) });
});
SidebarItem.displayName = "SidebarItem";
function ListItem({
  id,
  theme,
  isCollapsed,
  tooltipChildren,
  children: wrapperChildren,
  ...props
}) {
  return /* @__PURE__ */ jsxRuntime.jsx("li", { ...props, children: isCollapsed ? /* @__PURE__ */ jsxRuntime.jsx(
    Tooltip.Tooltip,
    {
      content: /* @__PURE__ */ jsxRuntime.jsx(Children, { id, theme, children: tooltipChildren }),
      placement: "right",
      children: wrapperChildren
    }
  ) : wrapperChildren });
}
ListItem.displayName = "SidebarItem.ListItem";
function Children({ id, theme, children }) {
  return /* @__PURE__ */ jsxRuntime.jsx(
    "span",
    {
      "data-testid": "flowbite-sidebar-item-content",
      id: `flowbite-sidebar-item-${id}`,
      className: tailwindMerge.twMerge(theme.content.base),
      children
    }
  );
}
ListItem.displayName = "SidebarItem.Children";

exports.SidebarItem = SidebarItem;
//# sourceMappingURL=SidebarItem.cjs.map
