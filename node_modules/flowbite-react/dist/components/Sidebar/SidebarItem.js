'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { forwardRef, useId } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Badge } from '../Badge/Badge.js';
import { Tooltip } from '../Tooltip/Tooltip.js';
import { useSidebarContext } from './SidebarContext.js';
import { useSidebarItemContext } from './SidebarItemContext.js';
import { sidebarTheme } from './theme.js';

const SidebarItem = forwardRef((props, ref) => {
  const id = useId();
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = useSidebarContext();
  const { isInsideCollapse } = useSidebarItemContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme.item, provider.theme?.sidebar?.item, rootTheme?.item, props.theme],
    [get(provider.clearTheme, "sidebar.item"), get(rootClearTheme, "item"), props.clearTheme],
    [get(provider.applyTheme, "sidebar.item"), get(rootApplyTheme, "item"), props.applyTheme]
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
  } = resolveProps(props, provider.props?.sidebarItem);
  return /* @__PURE__ */ jsx(ListItem, { theme, className: theme.listItem, id, isCollapsed, tooltipChildren: children, children: /* @__PURE__ */ jsxs(
    Component,
    {
      "aria-labelledby": `flowbite-sidebar-item-${id}`,
      ref,
      className: twMerge(
        theme.base,
        isActive && theme.active,
        !isCollapsed && isInsideCollapse && theme.collapsed.insideCollapse,
        className
      ),
      ...restProps,
      children: [
        Icon && /* @__PURE__ */ jsx(
          Icon,
          {
            "aria-hidden": true,
            "data-testid": "flowbite-sidebar-item-icon",
            className: twMerge(theme.icon.base, isActive && theme.icon.active)
          }
        ),
        isCollapsed && !Icon && /* @__PURE__ */ jsx("span", { className: theme.collapsed.noIcon, children: children.charAt(0).toLocaleUpperCase() ?? "?" }),
        !isCollapsed && /* @__PURE__ */ jsx(Children, { id, theme, children }),
        !isCollapsed && label && /* @__PURE__ */ jsx(Badge, { color: labelColor, "data-testid": "flowbite-sidebar-label", hidden: isCollapsed, className: theme.label, children: label })
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
  return /* @__PURE__ */ jsx("li", { ...props, children: isCollapsed ? /* @__PURE__ */ jsx(
    Tooltip,
    {
      content: /* @__PURE__ */ jsx(Children, { id, theme, children: tooltipChildren }),
      placement: "right",
      children: wrapperChildren
    }
  ) : wrapperChildren });
}
ListItem.displayName = "SidebarItem.ListItem";
function Children({ id, theme, children }) {
  return /* @__PURE__ */ jsx(
    "span",
    {
      "data-testid": "flowbite-sidebar-item-content",
      id: `flowbite-sidebar-item-${id}`,
      className: twMerge(theme.content.base),
      children
    }
  );
}
ListItem.displayName = "SidebarItem.Children";

export { SidebarItem };
//# sourceMappingURL=SidebarItem.js.map
