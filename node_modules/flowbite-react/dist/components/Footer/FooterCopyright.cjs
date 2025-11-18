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

const FooterCopyright = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.copyright, provider$1.theme?.footer?.copyright, props.theme],
    [get.get(provider$1.clearTheme, "footer.copyright"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.copyright"), props.applyTheme]
  );
  const { by, className, href, year, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerCopyright);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { ref, "data-testid": "flowbite-footer-copyright", className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: [
    "\xA9 ",
    year,
    href ? /* @__PURE__ */ jsxRuntime.jsx("a", { href, className: theme$1.href, children: by }) : /* @__PURE__ */ jsxRuntime.jsx("span", { "data-testid": "flowbite-footer-copyright-span", className: theme$1.span, children: by })
  ] });
});
FooterCopyright.displayName = "FooterCopyright";

exports.FooterCopyright = FooterCopyright;
//# sourceMappingURL=FooterCopyright.cjs.map
