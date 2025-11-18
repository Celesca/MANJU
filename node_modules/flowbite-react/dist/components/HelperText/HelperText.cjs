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

const HelperText = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.helperTextTheme, provider$1.theme?.helperText, props.theme],
    [get.get(provider$1.clearTheme, "helperText"), props.clearTheme],
    [get.get(provider$1.applyTheme, "helperText"), props.applyTheme]
  );
  const { className, color = "gray", ...restProps } = resolveProps.resolveProps(props, provider$1.props?.helperText);
  return /* @__PURE__ */ jsxRuntime.jsx("p", { ref, className: tailwindMerge.twMerge(theme$1.root.base, theme$1.root.colors[color], className), ...restProps });
});
HelperText.displayName = "HelperText";

exports.HelperText = HelperText;
//# sourceMappingURL=HelperText.cjs.map
