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
var SidebarItemContext = require('./SidebarItemContext.cjs');
var theme = require('./theme.cjs');

const SidebarItemGroup = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = SidebarContext.useSidebarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.sidebarTheme.itemGroup, provider$1.theme?.sidebar?.itemGroup, rootTheme?.itemGroup, props.theme],
    [get.get(provider$1.clearTheme, "sidebar.itemGroup"), get.get(rootClearTheme, "itemGroup"), props.clearTheme],
    [get.get(provider$1.applyTheme, "sidebar.itemGroup"), get.get(rootApplyTheme, "itemGroup"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.sidebarItemGroup);
  return /* @__PURE__ */ jsxRuntime.jsx(SidebarItemContext.SidebarItemContext.Provider, { value: { isInsideCollapse: false }, children: /* @__PURE__ */ jsxRuntime.jsx(
    "ul",
    {
      ref,
      "data-testid": "flowbite-sidebar-item-group",
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps
    }
  ) });
});
SidebarItemGroup.displayName = "SidebarItemGroup";

exports.SidebarItemGroup = SidebarItemGroup;
//# sourceMappingURL=SidebarItemGroup.cjs.map
