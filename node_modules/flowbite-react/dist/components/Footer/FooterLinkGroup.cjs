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

const FooterLinkGroup = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.groupLink, provider$1.theme?.footer?.groupLink, props.theme],
    [get.get(provider$1.clearTheme, "footer.groupLink"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.groupLink"), props.applyTheme]
  );
  const { className, col = false, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerLinkGroup);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "ul",
    {
      ref,
      "data-testid": "footer-groupLink",
      className: tailwindMerge.twMerge(theme$1.base, col && theme$1.col, className),
      ...restProps
    }
  );
});
FooterLinkGroup.displayName = "FooterLinkGroup";

exports.FooterLinkGroup = FooterLinkGroup;
//# sourceMappingURL=FooterLinkGroup.cjs.map
