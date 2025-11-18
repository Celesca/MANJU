'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var quoteRightIcon = require('../../icons/quote-right-icon.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const HRIcon = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.hrTheme.icon, provider$1.theme?.hr?.icon, props.theme],
    [get.get(provider$1.clearTheme, "hr.icon"), props.clearTheme],
    [get.get(provider$1.applyTheme, "hr.icon"), props.applyTheme]
  );
  const { icon: Icon = quoteRightIcon.QuoteRightIcon, className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.hrIcon);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.base, children: [
    /* @__PURE__ */ jsxRuntime.jsx(
      "hr",
      {
        ref,
        className: tailwindMerge.twMerge(theme$1.hrLine, className),
        "data-testid": "flowbite-hr-icon",
        role: "separator",
        ...restProps
      }
    ),
    /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.icon.base, children: /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, className: theme$1.icon.icon }) })
  ] });
});
HRIcon.displayName = "HRIcon";

exports.HRIcon = HRIcon;
//# sourceMappingURL=HRIcon.cjs.map
