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

const NavbarCollapse = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isOpen } = NavbarContext.useNavbarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.navbarTheme.collapse, provider$1.theme?.navbar?.collapse, rootTheme?.collapse, props.theme],
    [get.get(provider$1.clearTheme, "navbar.collapse"), get.get(rootClearTheme, "collapse"), props.clearTheme],
    [get.get(provider$1.applyTheme, "navbar.collapse"), get.get(rootApplyTheme, "collapse"), props.applyTheme]
  );
  const { children, className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.navbarCollapse);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "div",
    {
      ref,
      "data-testid": "flowbite-navbar-collapse",
      className: tailwindMerge.twMerge(theme$1.base, theme$1.hidden[!isOpen ? "on" : "off"], className),
      ...restProps,
      children: /* @__PURE__ */ jsxRuntime.jsx("ul", { className: theme$1.list, children })
    }
  );
});
NavbarCollapse.displayName = "NavbarCollapse";

exports.NavbarCollapse = NavbarCollapse;
//# sourceMappingURL=NavbarCollapse.cjs.map
