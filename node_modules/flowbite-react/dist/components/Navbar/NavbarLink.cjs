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

const NavbarLink = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, setIsOpen } = NavbarContext.useNavbarContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.navbarTheme.link, provider$1.theme?.navbar?.link, rootTheme?.link, props.theme],
    [get.get(provider$1.clearTheme, "navbar.link"), get.get(rootClearTheme, "link"), props.clearTheme],
    [get.get(provider$1.applyTheme, "navbar.link"), get.get(rootApplyTheme, "link"), props.applyTheme]
  );
  const {
    active,
    as: Component = "a",
    disabled,
    children,
    className,
    onClick,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.navbarLink);
  function handleClick(event) {
    setIsOpen(false);
    onClick?.(event);
  }
  return /* @__PURE__ */ jsxRuntime.jsx("li", { ref, children: /* @__PURE__ */ jsxRuntime.jsx(
    Component,
    {
      className: tailwindMerge.twMerge(
        theme$1.base,
        active && theme$1.active.on,
        !active && !disabled && theme$1.active.off,
        theme$1.disabled[disabled ? "on" : "off"],
        className
      ),
      onClick: handleClick,
      ...restProps,
      children
    }
  ) });
});
NavbarLink.displayName = "NavbarLink";

exports.NavbarLink = NavbarLink;
//# sourceMappingURL=NavbarLink.cjs.map
