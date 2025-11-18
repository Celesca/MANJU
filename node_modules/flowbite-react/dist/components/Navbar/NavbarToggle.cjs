'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var barsIcon = require('../../icons/bars-icon.cjs');
var provider = require('../../theme/provider.cjs');
var NavbarContext = require('./NavbarContext.cjs');
var theme = require('./theme.cjs');

const NavbarToggle = React.forwardRef((props, ref) => {
  const {
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme,
    isOpen,
    setIsOpen
  } = NavbarContext.useNavbarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.navbarTheme.toggle, provider$1.theme?.navbar?.toggle, rootTheme?.toggle, props.theme],
    [get.get(provider$1.clearTheme, "navbar.toggle"), get.get(rootClearTheme, "toggle"), props.clearTheme],
    [get.get(provider$1.applyTheme, "navbar.toggle"), get.get(rootApplyTheme, "toggle"), props.applyTheme]
  );
  const { barIcon: BarIcon = barsIcon.BarsIcon, className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.navbarToggle);
  function handleClick() {
    setIsOpen(!isOpen);
  }
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "button",
    {
      ref,
      "data-testid": "flowbite-navbar-toggle",
      onClick: handleClick,
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps,
      children: [
        /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.title, children: "Open main menu" }),
        /* @__PURE__ */ jsxRuntime.jsx(BarIcon, { "aria-hidden": true, className: theme$1.icon })
      ]
    }
  );
});
NavbarToggle.displayName = "NavbarToggle";

exports.NavbarToggle = NavbarToggle;
//# sourceMappingURL=NavbarToggle.cjs.map
