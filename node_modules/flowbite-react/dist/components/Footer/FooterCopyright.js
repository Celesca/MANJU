'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterCopyright = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.copyright, provider.theme?.footer?.copyright, props.theme],
    [get(provider.clearTheme, "footer.copyright"), props.clearTheme],
    [get(provider.applyTheme, "footer.copyright"), props.applyTheme]
  );
  const { by, className, href, year, ...restProps } = resolveProps(props, provider.props?.footerCopyright);
  return /* @__PURE__ */ jsxs("div", { ref, "data-testid": "flowbite-footer-copyright", className: twMerge(theme.base, className), ...restProps, children: [
    "\xA9 ",
    year,
    href ? /* @__PURE__ */ jsx("a", { href, className: theme.href, children: by }) : /* @__PURE__ */ jsx("span", { "data-testid": "flowbite-footer-copyright-span", className: theme.span, children: by })
  ] });
});
FooterCopyright.displayName = "FooterCopyright";

export { FooterCopyright };
//# sourceMappingURL=FooterCopyright.js.map
