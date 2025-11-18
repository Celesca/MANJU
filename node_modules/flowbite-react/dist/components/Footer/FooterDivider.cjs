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

const FooterDivider = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.divider, provider$1.theme?.footer?.divider, props.theme],
    [get.get(provider$1.clearTheme, "footer.divider"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.divider"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerDivider);
  return /* @__PURE__ */ jsxRuntime.jsx("hr", { ref, "data-testid": "footer-divider", className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
FooterDivider.displayName = "FooterDivider";

exports.FooterDivider = FooterDivider;
//# sourceMappingURL=FooterDivider.cjs.map
