'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var provider = require('../../theme/provider.cjs');
var Navbar = require('../Navbar/Navbar.cjs');
require('../Navbar/NavbarBrand.cjs');
require('../Navbar/NavbarCollapse.cjs');
require('../Navbar/NavbarContext.cjs');
require('../Navbar/NavbarLink.cjs');
require('../Navbar/NavbarToggle.cjs');
var theme = require('./theme.cjs');

const MegaMenu = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.megaMenuTheme, provider$1.theme?.megaMenu, props.theme],
    [get.get(provider$1.clearTheme, "megaMenu"), props.clearTheme],
    [get.get(provider$1.applyTheme, "megaMenu"), props.applyTheme]
  );
  const mergedProps = resolveProps.resolveProps(props, provider$1.props?.megaMenu);
  return /* @__PURE__ */ jsxRuntime.jsx(Navbar.Navbar, { ref, theme: theme$1, fluid: true, ...mergedProps });
});
MegaMenu.displayName = "MegaMenu";

exports.MegaMenu = MegaMenu;
//# sourceMappingURL=MegaMenu.cjs.map
