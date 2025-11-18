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

const HRTrimmed = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.hrTheme.trimmed, provider$1.theme?.hr?.trimmed, props.theme],
    [get.get(provider$1.clearTheme, "hr.trimmed"), props.clearTheme],
    [get.get(provider$1.applyTheme, "hr.trimmed"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.hrTrimmed);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "hr",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.base, className),
      "data-testid": "flowbite-hr-trimmed",
      role: "separator",
      ...restProps
    }
  );
});
HRTrimmed.displayName = "HRTrimmed";

exports.HRTrimmed = HRTrimmed;
//# sourceMappingURL=HRTrimmed.cjs.map
