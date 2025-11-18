'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterBrand = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.brand, provider.theme?.footer?.brand, props.theme],
    [get(provider.clearTheme, "footer.brand"), props.clearTheme],
    [get(provider.applyTheme, "footer.brand"), props.applyTheme]
  );
  const { alt, className, children, href, name, src, ...restProps } = resolveProps(props, provider.props?.footerBrand);
  return /* @__PURE__ */ jsx("div", { ref, children: href ? /* @__PURE__ */ jsxs("a", { "data-testid": "flowbite-footer-brand", href, className: twMerge(theme.base, className), ...restProps, children: [
    /* @__PURE__ */ jsx("img", { alt, src, className: theme.img }),
    /* @__PURE__ */ jsx("span", { "data-testid": "flowbite-footer-brand-span", className: theme.span, children: name }),
    children
  ] }) : /* @__PURE__ */ jsx(
    "img",
    {
      alt,
      "data-testid": "flowbite-footer-brand",
      src,
      className: twMerge(theme.img, className),
      ...restProps
    }
  ) });
});
FooterBrand.displayName = "FooterBrand";

export { FooterBrand };
//# sourceMappingURL=FooterBrand.js.map
