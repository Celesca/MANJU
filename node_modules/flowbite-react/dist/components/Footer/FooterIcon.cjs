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

const FooterIcon = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme.icon, provider$1.theme?.footer?.icon, props.theme],
    [get.get(provider$1.clearTheme, "footer.icon"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer.icon"), props.applyTheme]
  );
  const { ariaLabel, className, href, icon: Icon, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.footerIcon);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, children: href ? /* @__PURE__ */ jsxRuntime.jsx(
    "a",
    {
      "aria-label": ariaLabel,
      "data-testid": "flowbite-footer-icon",
      href,
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps,
      children: /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.size })
    }
  ) : /* @__PURE__ */ jsxRuntime.jsx(Icon, { "data-testid": "flowbite-footer-icon", className: theme$1.size, ...restProps }) });
});
FooterIcon.displayName = "FooterIcon";

exports.FooterIcon = FooterIcon;
//# sourceMappingURL=FooterIcon.cjs.map
