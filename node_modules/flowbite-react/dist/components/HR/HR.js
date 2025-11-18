'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { hrTheme } from './theme.js';

const HR = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [hrTheme.root, provider.theme?.hr?.root, props.theme],
    [get(provider.clearTheme, "hr.root"), props.clearTheme],
    [get(provider.applyTheme, "hr.root"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.hr);
  return /* @__PURE__ */ jsx(
    "hr",
    {
      ref,
      className: twMerge(theme.base, className),
      "data-testid": "flowbite-hr",
      role: "separator",
      ...restProps
    }
  );
});
HR.displayName = "HR";

export { HR };
//# sourceMappingURL=HR.js.map
