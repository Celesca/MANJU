'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { rangeSliderTheme } from './theme.js';

const RangeSlider = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [rangeSliderTheme, provider.theme?.rangeSlider, props.theme],
    [get(provider.clearTheme, "rangeSlider"), props.clearTheme],
    [get(provider.applyTheme, "rangeSlider"), props.applyTheme]
  );
  const { className, sizing = "md", ...restProps } = resolveProps(props, provider.props?.rangeSlider);
  return /* @__PURE__ */ jsx("div", { "data-testid": "flowbite-range-slider", className: twMerge(theme.root.base, className), children: /* @__PURE__ */ jsx("div", { className: theme.field.base, children: /* @__PURE__ */ jsx(
    "input",
    {
      ref,
      type: "range",
      className: twMerge(theme.field.input.base, theme.field.input.sizes[sizing]),
      ...restProps
    }
  ) }) });
});
RangeSlider.displayName = "RangeSlider";

export { RangeSlider };
//# sourceMappingURL=RangeSlider.js.map
