'use client';
import { jsx, jsxs, Fragment } from 'react/jsx-runtime';
import { forwardRef, useId, useState, useEffect } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { ChevronDownIcon } from '../../icons/chevron-down-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Tooltip } from '../Tooltip/Tooltip.js';
import { useSidebarContext } from './SidebarContext.js';
import { SidebarItemContext } from './SidebarItemContext.js';
import { sidebarTheme } from './theme.js';

const SidebarCollapse = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = useSidebarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme.collapse, provider.theme?.sidebar?.collapse, rootTheme?.collapse, props.theme],
    [get(provider.clearTheme, "sidebar.collapse"), get(rootClearTheme, "collapse"), props.clearTheme],
    [get(provider.applyTheme, "sidebar.collapse"), get(rootApplyTheme, "collapse"), props.applyTheme]
  );
  const {
    children,
    className,
    icon: Icon,
    label,
    chevronIcon: ChevronIcon = ChevronDownIcon,
    renderChevronIcon,
    open = false,
    ...restProps
  } = resolveProps(props, provider.props?.sidebarCollapse);
  const id = useId();
  const [isOpen, setOpen] = useState(open);
  useEffect(() => setOpen(open), [open]);
  function Wrapper({ children: children2 }) {
    if (isCollapsed && !isOpen) {
      return /* @__PURE__ */ jsx(Tooltip, { content: label, placement: "right", children: children2 });
    }
    return children2;
  }
  return /* @__PURE__ */ jsx("li", { ref, children: /* @__PURE__ */ jsxs(Wrapper, { children: [
    /* @__PURE__ */ jsxs(
      "button",
      {
        id: `flowbite-sidebar-collapse-${id}`,
        onClick: () => setOpen(!isOpen),
        title: label,
        type: "button",
        className: twMerge(theme.button, className),
        ...restProps,
        children: [
          Icon && /* @__PURE__ */ jsx(
            Icon,
            {
              "aria-hidden": true,
              "data-testid": "flowbite-sidebar-collapse-icon",
              className: twMerge(theme.icon.base, theme.icon.open[isOpen ? "on" : "off"])
            }
          ),
          isCollapsed ? /* @__PURE__ */ jsx("span", { className: theme.label.title, children: label }) : /* @__PURE__ */ jsxs(Fragment, { children: [
            /* @__PURE__ */ jsx("span", { "data-testid": "flowbite-sidebar-collapse-label", className: theme.label.base, children: label }),
            renderChevronIcon ? renderChevronIcon(theme, isOpen) : /* @__PURE__ */ jsx(
              ChevronIcon,
              {
                "aria-hidden": true,
                className: twMerge(theme.label.icon.base, theme.label.icon.open[isOpen ? "on" : "off"])
              }
            )
          ] })
        ]
      }
    ),
    /* @__PURE__ */ jsx("ul", { "aria-labelledby": `flowbite-sidebar-collapse-${id}`, hidden: !isOpen, className: theme.list, children: /* @__PURE__ */ jsx(SidebarItemContext.Provider, { value: { isInsideCollapse: true }, children }) })
  ] }) });
});
SidebarCollapse.displayName = "SidebarCollapse";

export { SidebarCollapse };
//# sourceMappingURL=SidebarCollapse.js.map
