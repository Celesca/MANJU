'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var ButtonBase = require('./ButtonBase.cjs');
var ButtonGroupContext = require('./ButtonGroupContext.cjs');
var theme = require('./theme.cjs');

const Button = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.buttonTheme, provider$1.theme?.button, props.theme],
    [get.get(provider$1.clearTheme, "button"), props.clearTheme],
    [get.get(provider$1.applyTheme, "button"), props.applyTheme]
  );
  const {
    children,
    className,
    color = "default",
    disabled,
    fullSized,
    outline: _outline,
    pill: _pill,
    size = "md",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.button);
  const buttonGroup = ButtonGroupContext.useButtonGroupContext();
  const outline = buttonGroup?.outline ?? _outline;
  const pill = buttonGroup?.pill ?? _pill;
  return /* @__PURE__ */ jsxRuntime.jsx(
    ButtonBase.ButtonBase,
    {
      ref,
      disabled,
      className: tailwindMerge.twMerge(
        theme$1.base,
        theme$1.size[size],
        pill && theme$1.pill,
        disabled && theme$1.disabled,
        fullSized && theme$1.fullSized,
        outline ? theme$1.outlineColor[color] : theme$1.color[color],
        buttonGroup && theme$1.grouped,
        className
      ),
      ...restProps,
      children
    }
  );
});
Button.displayName = "Button";

exports.Button = Button;
//# sourceMappingURL=Button.cjs.map
