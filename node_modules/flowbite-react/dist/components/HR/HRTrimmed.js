'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { hrTheme } from './theme.js';

const HRTrimmed = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [hrTheme.trimmed, provider.theme?.hr?.trimmed, props.theme],
    [get(provider.clearTheme, "hr.trimmed"), props.clearTheme],
    [get(provider.applyTheme, "hr.trimmed"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.hrTrimmed);
  return /* @__PURE__ */ jsx(
    "hr",
    {
      ref,
      className: twMerge(theme.base, className),
      "data-testid": "flowbite-hr-trimmed",
      role: "separator",
      ...restProps
    }
  );
});
HRTrimmed.displayName = "HRTrimmed";

export { HRTrimmed };
//# sourceMappingURL=HRTrimmed.js.map
