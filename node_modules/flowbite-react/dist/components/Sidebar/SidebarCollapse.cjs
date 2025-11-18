'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var chevronDownIcon = require('../../icons/chevron-down-icon.cjs');
var provider = require('../../theme/provider.cjs');
var Tooltip = require('../Tooltip/Tooltip.cjs');
var SidebarContext = require('./SidebarContext.cjs');
var SidebarItemContext = require('./SidebarItemContext.cjs');
var theme = require('./theme.cjs');

const SidebarCollapse = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = SidebarContext.useSidebarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme.collapse, provider$1.theme?.sidebar?.collapse, rootTheme?.collapse, props.theme],
    [get.get(provider$1.clearTheme, "sidebar.collapse"), get.get(rootClearTheme, "collapse"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar.collapse"), get.get(rootApplyTheme, "collapse"), props.applyTheme]
  );
  const {
    children,
    className,
    icon: Icon,
    label,
    chevronIcon: ChevronIcon = chevronDownIcon.ChevronDownIcon,
    renderChevronIcon,
    open = false,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.sidebarCollapse);
  const id = React.useId();
  const [isOpen, setOpen] = React.useState(open);
  React.useEffect(() => setOpen(open), [open]);
  function Wrapper({ children: children2 }) {
    if (isCollapsed && !isOpen) {
      return /* @__PURE__ */ jsxRuntime.jsx(Tooltip.Tooltip, { content: label, placement: "right", children: children2 });
    }
    return children2;
  }
  return /* @__PURE__ */ jsxRuntime.jsx("li", { ref, children: /* @__PURE__ */ jsxRuntime.jsxs(Wrapper, { children: [
    /* @__PURE__ */ jsxRuntime.jsxs(
      "button",
      {
        id: `flowbite-sidebar-collapse-${id}`,
        onClick: () => setOpen(!isOpen),
        title: label,
        type: "button",
        className: tailwindMerge.twMerge(theme$1.button, className),
        ...restProps,
        children: [
          Icon && /* @__PURE__ */ jsxRuntime.jsx(
            Icon,
            {
              "aria-hidden": true,
              "data-testid": "flowbite-sidebar-collapse-icon",
              className: tailwindMerge.twMerge(theme$1.icon.base, theme$1.icon.open[isOpen ? "on" : "off"])
            }
          ),
          isCollapsed ? /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.label.title, children: label }) : /* @__PURE__ */ jsxRuntime.jsxs(jsxRuntime.Fragment, { children: [
            /* @__PURE__ */ jsxRuntime.jsx("span", { "data-testid": "flowbite-sidebar-collapse-label", className: theme$1.label.base, children: label }),
            renderChevronIcon ? renderChevronIcon(theme$1, isOpen) : /* @__PURE__ */ jsxRuntime.jsx(
              ChevronIcon,
              {
                "aria-hidden": true,
                className: tailwindMerge.twMerge(theme$1.label.icon.base, theme$1.label.icon.open[isOpen ? "on" : "off"])
              }
            )
          ] })
        ]
      }
    ),
    /* @__PURE__ */ jsxRuntime.jsx("ul", { "aria-labelledby": `flowbite-sidebar-collapse-${id}`, hidden: !isOpen, className: theme$1.list, children: /* @__PURE__ */ jsxRuntime.jsx(SidebarItemContext.SidebarItemContext.Provider, { value: { isInsideCollapse: true }, children }) })
  ] }) });
});
SidebarCollapse.displayName = "SidebarCollapse";

exports.SidebarCollapse = SidebarCollapse;
//# sourceMappingURL=SidebarCollapse.cjs.map
