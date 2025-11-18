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

const SidebarCTA = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = SidebarContext.useSidebarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme.cta, provider$1.theme?.sidebar?.cta, rootTheme?.cta, props.theme],
    [get.get(provider$1.clearTheme, "sidebar.cta"), get.get(rootClearTheme, "cta"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar.cta"), get.get(rootApplyTheme, "cta"), props.applyTheme]
  );
  const { color = "info", className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.sidebarCTA);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "div",
    {
      ref,
      "data-testid": "sidebar-cta",
      hidden: isCollapsed,
      className: tailwindMerge.twMerge(theme$1.base, theme$1.color[color], className),
      ...restProps
    }
  );
});
SidebarCTA.displayName = "SidebarCTA";

exports.SidebarCTA = SidebarCTA;
//# sourceMappingURL=SidebarCTA.cjs.map
