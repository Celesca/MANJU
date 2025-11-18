'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { ButtonGroupContext } from './ButtonGroupContext.js';
import { buttonGroupTheme } from './theme.js';

const ButtonGroup = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [buttonGroupTheme, provider.theme?.buttonGroup, props.theme],
    [get(provider.clearTheme, "buttonGroup"), props.clearTheme],
    [get(provider.applyTheme, "buttonGroup"), props.applyTheme]
  );
  const { children, className, outline, pill, ...restProps } = resolveProps(props, provider.props?.buttonGroup);
  return /* @__PURE__ */ jsx(ButtonGroupContext.Provider, { value: { outline, pill }, children: /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.base, className), role: "group", ...restProps, children }) });
});
ButtonGroup.displayName = "ButtonGroup";

export { ButtonGroup };
//# sourceMappingURL=ButtonGroup.js.map
