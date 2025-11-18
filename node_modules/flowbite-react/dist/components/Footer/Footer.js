'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const Footer = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme, provider.theme?.footer, props.theme],
    [get(provider.clearTheme, "footer"), props.clearTheme],
    [get(provider.applyTheme, "footer"), props.applyTheme]
  );
  const {
    bgDark = false,
    children,
    className,
    container = false,
    ...restProps
  } = resolveProps(props, provider.props?.footer);
  return /* @__PURE__ */ jsx(
    "footer",
    {
      ref,
      "data-testid": "flowbite-footer",
      className: twMerge(theme.root.base, bgDark && theme.root.bgDark, container && theme.root.container, className),
      ...restProps,
      children
    }
  );
});
Footer.displayName = "Footer";

export { Footer };
//# sourceMappingURL=Footer.js.map
