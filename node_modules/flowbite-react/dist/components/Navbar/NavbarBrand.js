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

const NavbarBrand = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useNavbarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [navbarTheme.brand, provider.theme?.navbar?.brand, rootTheme?.brand, props.theme],
    [get(provider.clearTheme, "navbar.brand"), get(rootClearTheme, "brand"), props.clearTheme],
    [get(provider.applyTheme, "navbar.brand"), get(rootApplyTheme, "brand"), props.applyTheme]
  );
  const { as: Component = "a", className, ...restProps } = resolveProps(props, provider.props?.navbarBrand);
  return /* @__PURE__ */ jsx(Component, { ref, className: twMerge(theme.base, className), ...restProps });
});
NavbarBrand.displayName = "NavbarBrand";

export { NavbarBrand };
//# sourceMappingURL=NavbarBrand.js.map
