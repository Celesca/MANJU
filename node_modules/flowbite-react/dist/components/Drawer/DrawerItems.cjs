'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var DrawerContext = require('./DrawerContext.cjs');
var theme = require('./theme.cjs');

const DrawerItems = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = DrawerContext.useDrawerContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.drawerTheme.items, provider$1.theme?.drawer?.items, rootTheme?.items, props.theme],
    [get.get(provider$1.clearTheme, "drawer.items"), get.get(rootClearTheme, "items"), props.clearTheme],
    [get.get(provider$1.applyTheme, "drawer.items"), get.get(rootApplyTheme, "items"), props.applyTheme]
  );
  const { children, className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.drawerItems);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, "data-testid": "flowbite-drawer-items", className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children });
});
DrawerItems.displayName = "DrawerItems";

exports.DrawerItems = DrawerItems;
//# sourceMappingURL=DrawerItems.cjs.map
