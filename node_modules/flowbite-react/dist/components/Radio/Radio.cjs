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

const Radio = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.radioTheme, provider$1.theme?.radio, props.theme],
    [get.get(provider$1.clearTheme, "radio"), props.clearTheme],
    [get.get(provider$1.applyTheme, "radio"), props.applyTheme]
  );
  const { color = "default", className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.radio);
  return /* @__PURE__ */ jsxRuntime.jsx("input", { ref, type: "radio", className: tailwindMerge.twMerge(theme$1.base, theme$1.color[color], className), ...restProps });
});
Radio.displayName = "Radio";

exports.Radio = Radio;
//# sourceMappingURL=Radio.cjs.map
