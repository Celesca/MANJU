'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { textareaTheme } from './theme.js';

const Textarea = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [textareaTheme, provider.theme?.textarea, props.theme],
    [get(provider.clearTheme, "textarea"), props.clearTheme],
    [get(provider.applyTheme, "textarea"), props.applyTheme]
  );
  const { className, color = "gray", shadow, ...restProps } = resolveProps(props, provider.props?.textarea);
  return /* @__PURE__ */ jsx(
    "textarea",
    {
      ref,
      className: twMerge(theme.base, theme.colors[color], theme.withShadow[shadow ? "on" : "off"], className),
      ...restProps
    }
  );
});
Textarea.displayName = "Textarea";

export { Textarea };
//# sourceMappingURL=Textarea.js.map
