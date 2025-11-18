'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { hrTheme } from './theme.js';

const HRSquare = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [hrTheme.square, provider.theme?.hr?.square, props.theme],
    [get(provider.clearTheme, "hr.square"), props.clearTheme],
    [get(provider.applyTheme, "hr.square"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.hrSquare);
  return /* @__PURE__ */ jsx(
    "hr",
    {
      ref,
      className: twMerge(theme.base, className),
      "data-testid": "flowbite-hr-square",
      role: "separator",
      ...restProps
    }
  );
});
HRSquare.displayName = "HRSquare";

export { HRSquare };
//# sourceMappingURL=HRSquare.js.map
