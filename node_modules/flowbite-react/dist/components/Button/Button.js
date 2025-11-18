'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { ButtonBase } from './ButtonBase.js';
import { useButtonGroupContext } from './ButtonGroupContext.js';
import { buttonTheme } from './theme.js';

const Button = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [buttonTheme, provider.theme?.button, props.theme],
    [get(provider.clearTheme, "button"), props.clearTheme],
    [get(provider.applyTheme, "button"), props.applyTheme]
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
  } = resolveProps(props, provider.props?.button);
  const buttonGroup = useButtonGroupContext();
  const outline = buttonGroup?.outline ?? _outline;
  const pill = buttonGroup?.pill ?? _pill;
  return /* @__PURE__ */ jsx(
    ButtonBase,
    {
      ref,
      disabled,
      className: twMerge(
        theme.base,
        theme.size[size],
        pill && theme.pill,
        disabled && theme.disabled,
        fullSized && theme.fullSized,
        outline ? theme.outlineColor[color] : theme.color[color],
        buttonGroup && theme.grouped,
        className
      ),
      ...restProps,
      children
    }
  );
});
Button.displayName = "Button";

export { Button };
//# sourceMappingURL=Button.js.map
