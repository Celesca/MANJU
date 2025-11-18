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

const Navbar = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.navbarTheme, provider$1.theme?.navbar, props.theme],
    [get.get(provider$1.clearTheme, "navbar"), props.clearTheme],
    [get.get(provider$1.applyTheme, "navbar"), props.applyTheme]
  );
  const {
    border,
    children,
    className,
    fluid = false,
    menuOpen,
    rounded,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.navbar);
  const [isOpen, setIsOpen] = React.useState(menuOpen);
  return /* @__PURE__ */ jsxRuntime.jsx(
    NavbarContext.NavbarContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, isOpen, setIsOpen },
      children: /* @__PURE__ */ jsxRuntime.jsx(
        "nav",
        {
          ref,
          className: tailwindMerge.twMerge(
            theme$1.root.base,
            theme$1.root.bordered[border ? "on" : "off"],
            theme$1.root.rounded[rounded ? "on" : "off"],
            className
          ),
          ...restProps,
          children: /* @__PURE__ */ jsxRuntime.jsx("div", { className: tailwindMerge.twMerge(theme$1.root.inner.base, theme$1.root.inner.fluid[fluid ? "on" : "off"]), children })
        }
      )
    }
  );
});
Navbar.displayName = "Navbar";

exports.Navbar = Navbar;
//# sourceMappingURL=Navbar.cjs.map
