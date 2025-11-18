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

const Checkbox = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.checkboxTheme, provider$1.theme?.checkbox, props.theme],
    [get.get(provider$1.clearTheme, "checkbox"), props.clearTheme],
    [get.get(provider$1.applyTheme, "checkbox"), props.applyTheme]
  );
  const { className, color = "default", indeterminate, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.checkbox);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "input",
    {
      ref,
      type: "checkbox",
      className: tailwindMerge.twMerge(theme$1.base, theme$1.color[color], indeterminate && theme$1.indeterminate, className),
      ...restProps
    }
  );
});
Checkbox.displayName = "Checkbox";

exports.Checkbox = Checkbox;
//# sourceMappingURL=Checkbox.cjs.map
