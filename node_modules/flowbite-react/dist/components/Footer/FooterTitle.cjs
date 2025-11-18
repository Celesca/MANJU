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

const FooterTitle = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.title, provider$1.theme?.footer?.title, props.theme],
    [get.get(provider$1.clearTheme, "footer.title"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.title"), props.applyTheme]
  );
  const { as: Component = "h2", className, title, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerTitle);
  return /* @__PURE__ */ jsxRuntime.jsx(Component, { ref, "data-testid": "flowbite-footer-title", className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: title });
});
FooterTitle.displayName = "FooterTitle";

exports.FooterTitle = FooterTitle;
//# sourceMappingURL=FooterTitle.cjs.map
