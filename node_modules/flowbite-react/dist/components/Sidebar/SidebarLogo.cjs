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

const SidebarLogo = React.forwardRef((props, ref) => {
  const id = React.useId();
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = SidebarContext.useSidebarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme.logo, provider$1.theme?.sidebar?.logo, rootTheme?.logo, props.theme],
    [get.get(provider$1.clearTheme, "sidebar.logo"), get.get(rootClearTheme, "logo"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar.logo"), get.get(rootApplyTheme, "logo"), props.applyTheme]
  );
  const {
    children,
    className,
    href,
    img,
    imgAlt = "",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.sidebarLogo);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "a",
    {
      ref,
      "aria-labelledby": `flowbite-sidebar-logo-${id}`,
      href,
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps,
      children: [
        /* @__PURE__ */ jsxRuntime.jsx("img", { alt: imgAlt, src: img, className: theme$1.img }),
        /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.collapsed[isCollapsed ? "on" : "off"], id: `flowbite-sidebar-logo-${id}`, children })
      ]
    }
  );
});
SidebarLogo.displayName = "SidebarLogo";

exports.SidebarLogo = SidebarLogo;
//# sourceMappingURL=SidebarLogo.cjs.map
