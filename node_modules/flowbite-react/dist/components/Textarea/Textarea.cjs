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

const Textarea = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.textareaTheme, provider$1.theme?.textarea, props.theme],
    [get.get(provider$1.clearTheme, "textarea"), props.clearTheme],
    [get.get(provider$1.applyTheme, "textarea"), props.applyTheme]
  );
  const { className, color = "gray", shadow, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.textarea);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "textarea",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.base, theme$1.colors[color], theme$1.withShadow[shadow ? "on" : "off"], className),
      ...restProps
    }
  );
});
Textarea.displayName = "Textarea";

exports.Textarea = Textarea;
//# sourceMappingURL=Textarea.cjs.map
