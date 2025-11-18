'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var ButtonGroupContext = require('./ButtonGroupContext.cjs');
var theme = require('./theme.cjs');

const ButtonGroup = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.buttonGroupTheme, provider$1.theme?.buttonGroup, props.theme],
    [get.get(provider$1.clearTheme, "buttonGroup"), props.clearTheme],
    [get.get(provider$1.applyTheme, "buttonGroup"), props.applyTheme]
  );
  const { children, className, outline, pill, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.buttonGroup);
  return /* @__PURE__ */ jsxRuntime.jsx(ButtonGroupContext.ButtonGroupContext.Provider, { value: { outline, pill }, children: /* @__PURE__ */ jsxRuntime.jsx("div", { ref, className: tailwindMerge.twMerge(theme$1.base, className), role: "group", ...restProps, children }) });
});
ButtonGroup.displayName = "ButtonGroup";

exports.ButtonGroup = ButtonGroup;
//# sourceMappingURL=ButtonGroup.cjs.map
