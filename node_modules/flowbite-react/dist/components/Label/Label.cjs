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

const Label = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.labelTheme, provider$1.theme?.label, props.theme],
    [get.get(provider$1.clearTheme, "label"), props.clearTheme],
    [get.get(provider$1.applyTheme, "label"), props.applyTheme]
  );
  const { className, color = "default", disabled = false, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.label);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "label",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.root.base, theme$1.root.colors[color], disabled && theme$1.root.disabled, className),
      "data-testid": "flowbite-label",
      ...restProps
    }
  );
});
Label.displayName = "Label";

exports.Label = Label;
//# sourceMappingURL=Label.cjs.map
