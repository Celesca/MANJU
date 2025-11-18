'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterLink = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.groupLink.link, provider.theme?.footer?.groupLink?.link, props.theme],
    [get(provider.clearTheme, "footer.groupLink.link"), props.clearTheme],
    [get(provider.applyTheme, "footer.groupLink.link"), props.applyTheme]
  );
  const { as: Component = "a", className, href, ...restProps } = resolveProps(props, provider.props?.footerLink);
  return /* @__PURE__ */ jsx("li", { ref, className: twMerge(theme.base, className), children: /* @__PURE__ */ jsx(Component, { href, className: theme.href, ...restProps }) });
});
FooterLink.displayName = "FooterLink";

export { FooterLink };
//# sourceMappingURL=FooterLink.js.map
