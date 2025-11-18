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

const HR = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.hrTheme.root, provider$1.theme?.hr?.root, props.theme],
    [get.get(provider$1.clearTheme, "hr.root"), props.clearTheme],
    [get.get(provider$1.applyTheme, "hr.root"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.hr);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "hr",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.base, className),
      "data-testid": "flowbite-hr",
      role: "separator",
      ...restProps
    }
  );
});
HR.displayName = "HR";

exports.HR = HR;
//# sourceMappingURL=HR.cjs.map
