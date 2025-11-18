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

const NavbarLink = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, setIsOpen } = useNavbarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [navbarTheme.link, provider.theme?.navbar?.link, rootTheme?.link, props.theme],
    [get(provider.clearTheme, "navbar.link"), get(rootClearTheme, "link"), props.clearTheme],
    [get(provider.applyTheme, "navbar.link"), get(rootApplyTheme, "link"), props.applyTheme]
  );
  const {
    active,
    as: Component = "a",
    disabled,
    children,
    className,
    onClick,
    ...restProps
  } = resolveProps(props, provider.props?.navbarLink);
  function handleClick(event) {
    setIsOpen(false);
    onClick?.(event);
  }
  return /* @__PURE__ */ jsx("li", { ref, children: /* @__PURE__ */ jsx(
    Component,
    {
      className: twMerge(
        theme.base,
        active && theme.active.on,
        !active && !disabled && theme.active.off,
        theme.disabled[disabled ? "on" : "off"],
        className
      ),
      onClick: handleClick,
      ...restProps,
      children
    }
  ) });
});
NavbarLink.displayName = "NavbarLink";

export { NavbarLink };
//# sourceMappingURL=NavbarLink.js.map
