'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { labelTheme } from './theme.js';

const Label = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [labelTheme, provider.theme?.label, props.theme],
    [get(provider.clearTheme, "label"), props.clearTheme],
    [get(provider.applyTheme, "label"), props.applyTheme]
  );
  const { className, color = "default", disabled = false, ...restProps } = resolveProps(props, provider.props?.label);
  return /* @__PURE__ */ jsx(
    "label",
    {
      ref,
      className: twMerge(theme.root.base, theme.root.colors[color], disabled && theme.root.disabled, className),
      "data-testid": "flowbite-label",
      ...restProps
    }
  );
});
Label.displayName = "Label";

export { Label };
//# sourceMappingURL=Label.js.map
