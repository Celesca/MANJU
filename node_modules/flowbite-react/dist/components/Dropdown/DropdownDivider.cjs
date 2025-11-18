'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var DropdownContext = require('./DropdownContext.cjs');
var theme = require('./theme.cjs');

const DropdownDivider = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = DropdownContext.useDropdownContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.dropdownTheme.floating, provider$1.theme?.dropdown?.floating, rootTheme?.floating, props.theme],
    [get.get(provider$1.clearTheme, "dropdown.floating"), get.get(rootClearTheme, "floating"), props.clearTheme],
    [get.get(provider$1.applyTheme, "dropdown.floating"), get.get(rootApplyTheme, "floating"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.dropdownDivider);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, className: tailwindMerge.twMerge(theme$1.divider, className), ...restProps });
});
DropdownDivider.displayName = "DropdownDivider";

exports.DropdownDivider = DropdownDivider;
//# sourceMappingURL=DropdownDivider.cjs.map
