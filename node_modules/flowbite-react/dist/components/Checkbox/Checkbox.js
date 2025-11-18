'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { checkboxTheme } from './theme.js';

const Checkbox = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [checkboxTheme, provider.theme?.checkbox, props.theme],
    [get(provider.clearTheme, "checkbox"), props.clearTheme],
    [get(provider.applyTheme, "checkbox"), props.applyTheme]
  );
  const { className, color = "default", indeterminate, ...restProps } = resolveProps(props, provider.props?.checkbox);
  return /* @__PURE__ */ jsx(
    "input",
    {
      ref,
      type: "checkbox",
      className: twMerge(theme.base, theme.color[color], indeterminate && theme.indeterminate, className),
      ...restProps
    }
  );
});
Checkbox.displayName = "Checkbox";

export { Checkbox };
//# sourceMappingURL=Checkbox.js.map
