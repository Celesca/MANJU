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

const SidebarItems = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = SidebarContext.useSidebarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme.items, provider$1.theme?.sidebar?.items, rootTheme?.items, props.theme],
    [get.get(provider$1.clearTheme, "sidebar.items"), get.get(rootClearTheme, "items"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar.items"), get.get(rootApplyTheme, "items"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.sidebarItems);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, className: tailwindMerge.twMerge(theme$1.base, className), "data-testid": "flowbite-sidebar-items", ...restProps });
});
SidebarItems.displayName = "SidebarItems";

exports.SidebarItems = SidebarItems;
//# sourceMappingURL=SidebarItems.cjs.map
