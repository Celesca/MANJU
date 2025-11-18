'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef, useState } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { NavbarContext } from './NavbarContext.js';
import { navbarTheme } from './theme.js';

const Navbar = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [navbarTheme, provider.theme?.navbar, props.theme],
    [get(provider.clearTheme, "navbar"), props.clearTheme],
    [get(provider.applyTheme, "navbar"), props.applyTheme]
  );
  const {
    border,
    children,
    className,
    fluid = false,
    menuOpen,
    rounded,
    ...restProps
  } = resolveProps(props, provider.props?.navbar);
  const [isOpen, setIsOpen] = useState(menuOpen);
  return /* @__PURE__ */ jsx(
    NavbarContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, isOpen, setIsOpen },
      children: /* @__PURE__ */ jsx(
        "nav",
        {
          ref,
          className: twMerge(
            theme.root.base,
            theme.root.bordered[border ? "on" : "off"],
            theme.root.rounded[rounded ? "on" : "off"],
            className
          ),
          ...restProps,
          children: /* @__PURE__ */ jsx("div", { className: twMerge(theme.root.inner.base, theme.root.inner.fluid[fluid ? "on" : "off"]), children })
        }
      )
    }
  );
});
Navbar.displayName = "Navbar";

export { Navbar };
//# sourceMappingURL=Navbar.js.map
