'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const FooterBrand = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.brand, provider$1.theme?.footer?.brand, props.theme],
    [get.get(provider$1.clearTheme, "footer.brand"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.brand"), props.applyTheme]
  );
  const { alt, className, children, href, name, src, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerBrand);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, children: href ? /* @__PURE__ */ jsxRuntime.jsxs("a", { "data-testid": "flowbite-footer-brand", href, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: [
    /* @__PURE__ */ jsxRuntime.jsx("img", { alt, src, className: theme$1.img }),
    /* @__PURE__ */ jsxRuntime.jsx("span", { "data-testid": "flowbite-footer-brand-span", className: theme$1.span, children: name }),
    children
  ] }) : /* @__PURE__ */ jsxRuntime.jsx(
    "img",
    {
      alt,
      "data-testid": "flowbite-footer-brand",
      src,
      className: tailwindMerge.twMerge(theme$1.img, className),
      ...restProps
    }
  ) });
});
FooterBrand.displayName = "FooterBrand";

exports.FooterBrand = FooterBrand;
//# sourceMappingURL=FooterBrand.cjs.map
