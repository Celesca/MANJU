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

const Footer = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.footerTheme, provider$1.theme?.footer, props.theme],
    [get.get(provider$1.clearTheme, "footer"), props.clearTheme],
    [get.get(provider$1.applyTheme, "footer"), props.applyTheme]
  );
  const {
    bgDark = false,
    children,
    className,
    container = false,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.footer);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "footer",
    {
      ref,
      "data-testid": "flowbite-footer",
      className: tailwindMerge.twMerge(theme$1.root.base, bgDark && theme$1.root.bgDark, container && theme$1.root.container, className),
      ...restProps,
      children
    }
  );
});
Footer.displayName = "Footer";

exports.Footer = Footer;
//# sourceMappingURL=Footer.cjs.map
