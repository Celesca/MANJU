'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { BarsIcon } from '../../icons/bars-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useNavbarContext } from './NavbarContext.js';
import { navbarTheme } from './theme.js';

const NavbarToggle = forwardRef((props, ref) => {
  const {
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme,
    isOpen,
    setIsOpen
  } = useNavbarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [navbarTheme.toggle, provider.theme?.navbar?.toggle, rootTheme?.toggle, props.theme],
    [get(provider.clearTheme, "navbar.toggle"), get(rootClearTheme, "toggle"), props.clearTheme],
    [get(provider.applyTheme, "navbar.toggle"), get(rootApplyTheme, "toggle"), props.applyTheme]
  );
  const { barIcon: BarIcon = BarsIcon, className, ...restProps } = resolveProps(props, provider.props?.navbarToggle);
  function handleClick() {
    setIsOpen(!isOpen);
  }
  return /* @__PURE__ */ jsxs(
    "button",
    {
      ref,
      "data-testid": "flowbite-navbar-toggle",
      onClick: handleClick,
      className: twMerge(theme.base, className),
      ...restProps,
      children: [
        /* @__PURE__ */ jsx("span", { className: theme.title, children: "Open main menu" }),
        /* @__PURE__ */ jsx(BarIcon, { "aria-hidden": true, className: theme.icon })
      ]
    }
  );
});
NavbarToggle.displayName = "NavbarToggle";

export { NavbarToggle };
//# sourceMappingURL=NavbarToggle.js.map
