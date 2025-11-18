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

const FloatingLabel = React.forwardRef((props, ref) => {
  const randomId = React.useId();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.floatingLabelTheme, provider$1.theme?.floatingLabel, props.theme],
    [get.get(provider$1.clearTheme, "floatingLabel"), props.clearTheme],
    [get.get(provider$1.applyTheme, "floatingLabel"), props.applyTheme]
  );
  const {
    label,
    color = "default",
    sizing = "md",
    variant,
    disabled = false,
    className,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.floatingLabel);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { className: tailwindMerge.twMerge("relative", variant === "standard" ? "z-0" : ""), children: [
    /* @__PURE__ */ jsxRuntime.jsx(
      "input",
      {
        type: "text",
        id: "floatingLabel" + randomId,
        "aria-describedby": "outlined_success_help",
        className: tailwindMerge.twMerge(theme$1.input[color][variant][sizing], className),
        placeholder: " ",
        "data-testid": "floating-label",
        disabled,
        ...restProps,
        ref
      }
    ),
    /* @__PURE__ */ jsxRuntime.jsx("label", { htmlFor: "floatingLabel" + randomId, className: tailwindMerge.twMerge(theme$1.label[color][variant][sizing], className), children: label })
  ] });
});
FloatingLabel.displayName = "FloatingLabel";

exports.FloatingLabel = FloatingLabel;
//# sourceMappingURL=FloatingLabel.cjs.map
