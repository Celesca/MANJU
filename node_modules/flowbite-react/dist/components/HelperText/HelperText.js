'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { helperTextTheme } from './theme.js';

const HelperText = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [helperTextTheme, provider.theme?.helperText, props.theme],
    [get(provider.clearTheme, "helperText"), props.clearTheme],
    [get(provider.applyTheme, "helperText"), props.applyTheme]
  );
  const { className, color = "gray", ...restProps } = resolveProps(props, provider.props?.helperText);
  return /* @__PURE__ */ jsx("p", { ref, className: twMerge(theme.root.base, theme.root.colors[color], className), ...restProps });
});
HelperText.displayName = "HelperText";

export { HelperText };
//# sourceMappingURL=HelperText.js.map
