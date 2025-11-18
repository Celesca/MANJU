'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var NavbarContext = require('./NavbarContext.cjs');
var theme = require('./theme.cjs');

const NavbarBrand = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = NavbarContext.useNavbarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.navbarTheme.brand, provider$1.theme?.navbar?.brand, rootTheme?.brand, props.theme],
    [get.get(provider$1.clearTheme, "navbar.brand"), get.get(rootClearTheme, "brand"), props.clearTheme],
    [get.get(provider$1.applyTheme, "navbar.brand"), get.get(rootApplyTheme, "brand"), props.applyTheme]
  );
  const { as: Component = "a", className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.navbarBrand);
  return /* @__PURE__ */ jsxRuntime.jsx(Component, { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
NavbarBrand.displayName = "NavbarBrand";

exports.NavbarBrand = NavbarBrand;
//# sourceMappingURL=NavbarBrand.cjs.map
