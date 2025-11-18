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

const FooterLink = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.groupLink.link, provider$1.theme?.footer?.groupLink?.link, props.theme],
    [get.get(provider$1.clearTheme, "footer.groupLink.link"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.groupLink.link"), props.applyTheme]
  );
  const { as: Component = "a", className, href, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerLink);
  return /* @__PURE__ */ jsxRuntime.jsx("li", { ref, className: tailwindMerge.twMerge(theme$1.base, className), children: /* @__PURE__ */ jsxRuntime.jsx(Component, { href, className: theme$1.href, ...restProps }) });
});
FooterLink.displayName = "FooterLink";

exports.FooterLink = FooterLink;
//# sourceMappingURL=FooterLink.cjs.map
