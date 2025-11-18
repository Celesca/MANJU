'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useNavbarContext } from './NavbarContext.js';
import { navbarTheme } from './theme.js';

const NavbarCollapse = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isOpen } = useNavbarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [navbarTheme.collapse, provider.theme?.navbar?.collapse, rootTheme?.collapse, props.theme],
    [get(provider.clearTheme, "navbar.collapse"), get(rootClearTheme, "collapse"), props.clearTheme],
    [get(provider.applyTheme, "navbar.collapse"), get(rootApplyTheme, "collapse"), props.applyTheme]
  );
  const { children, className, ...restProps } = resolveProps(props, provider.props?.navbarCollapse);
  return /* @__PURE__ */ jsx(
    "div",
    {
      ref,
      "data-testid": "flowbite-navbar-collapse",
      className: twMerge(theme.base, theme.hidden[!isOpen ? "on" : "off"], className),
      ...restProps,
      children: /* @__PURE__ */ jsx("ul", { className: theme.list, children })
    }
  );
});
NavbarCollapse.displayName = "NavbarCollapse";

export { NavbarCollapse };
//# sourceMappingURL=NavbarCollapse.js.map
