'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var provider = require('../../theme/provider.cjs');
var Floating = require('../Floating/Floating.cjs');
var theme = require('./theme.cjs');

function Tooltip(props) {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tooltipTheme, provider$1.theme?.tooltip, props.theme],
    [get.get(provider$1.clearTheme, "tooltip"), props.clearTheme],
    [get.get(provider$1.applyTheme, "tooltip"), props.applyTheme]
  );
  const {
    animation = "duration-300",
    arrow = true,
    children,
    className,
    content,
    placement = "top",
    style = "dark",
    trigger = "hover",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.tooltip);
  return /* @__PURE__ */ jsxRuntime.jsx(
    Floating.Floating,
    {
      animation,
      arrow,
      content,
      placement,
      style,
      theme: theme$1,
      trigger,
      className,
      ...restProps,
      children
    }
  );
}
Tooltip.displayName = "Tooltip";

exports.Tooltip = Tooltip;
//# sourceMappingURL=Tooltip.cjs.map
