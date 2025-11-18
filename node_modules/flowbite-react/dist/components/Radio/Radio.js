'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { radioTheme } from './theme.js';

const Radio = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [radioTheme, provider.theme?.radio, props.theme],
    [get(provider.clearTheme, "radio"), props.clearTheme],
    [get(provider.applyTheme, "radio"), props.applyTheme]
  );
  const { color = "default", className, ...restProps } = resolveProps(props, provider.props?.radio);
  return /* @__PURE__ */ jsx("input", { ref, type: "radio", className: twMerge(theme.base, theme.color[color], className), ...restProps });
});
Radio.displayName = "Radio";

export { Radio };
//# sourceMappingURL=Radio.js.map
