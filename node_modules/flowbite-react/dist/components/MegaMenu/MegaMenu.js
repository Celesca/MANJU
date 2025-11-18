'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Navbar } from '../Navbar/Navbar.js';
import '../Navbar/NavbarBrand.js';
import '../Navbar/NavbarCollapse.js';
import '../Navbar/NavbarContext.js';
import '../Navbar/NavbarLink.js';
import '../Navbar/NavbarToggle.js';
import { megaMenuTheme } from './theme.js';

const MegaMenu = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [megaMenuTheme, provider.theme?.megaMenu, props.theme],
    [get(provider.clearTheme, "megaMenu"), props.clearTheme],
    [get(provider.applyTheme, "megaMenu"), props.applyTheme]
  );
  const mergedProps = resolveProps(props, provider.props?.megaMenu);
  return /* @__PURE__ */ jsx(Navbar, { ref, theme, fluid: true, ...mergedProps });
});
MegaMenu.displayName = "MegaMenu";

export { MegaMenu };
//# sourceMappingURL=MegaMenu.js.map
