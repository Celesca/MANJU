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

const RangeSlider = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.rangeSliderTheme, provider$1.theme?.rangeSlider, props.theme],
    [get.get(provider$1.clearTheme, "rangeSlider"), props.clearTheme],
    [get.get(provider$1.applyTheme, "rangeSlider"), props.applyTheme]
  );
  const { className, sizing = "md", ...restProps } = resolveProps.resolveProps(props, provider$1.props?.rangeSlider);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { "data-testid": "flowbite-range-slider", className: tailwindMerge.twMerge(theme$1.root.base, className), children: /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.field.base, children: /* @__PURE__ */ jsxRuntime.jsx(
    "input",
    {
      ref,
      type: "range",
      className: tailwindMerge.twMerge(theme$1.field.input.base, theme$1.field.input.sizes[sizing]),
      ...restProps
    }
  ) }) });
});
RangeSlider.displayName = "RangeSlider";

exports.RangeSlider = RangeSlider;
//# sourceMappingURL=RangeSlider.cjs.map
