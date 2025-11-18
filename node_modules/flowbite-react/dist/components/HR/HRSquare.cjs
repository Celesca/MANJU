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

const HRSquare = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.hrTheme.square, provider$1.theme?.hr?.square, props.theme],
    [get.get(provider$1.clearTheme, "hr.square"), props.clearTheme],
    [get.get(provider$1.applyTheme, "hr.square"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.hrSquare);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "hr",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.base, className),
      "data-testid": "flowbite-hr-square",
      role: "separator",
      ...restProps
    }
  );
});
HRSquare.displayName = "HRSquare";

exports.HRSquare = HRSquare;
//# sourceMappingURL=HRSquare.cjs.map
