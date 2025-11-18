'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useDropdownContext } from './DropdownContext.js';
import { dropdownTheme } from './theme.js';

const DropdownDivider = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useDropdownContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [dropdownTheme.floating, provider.theme?.dropdown?.floating, rootTheme?.floating, props.theme],
    [get(provider.clearTheme, "dropdown.floating"), get(rootClearTheme, "floating"), props.clearTheme],
    [get(provider.applyTheme, "dropdown.floating"), get(rootApplyTheme, "floating"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.dropdownDivider);
  return /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.divider, className), ...restProps });
});
DropdownDivider.displayName = "DropdownDivider";

export { DropdownDivider };
//# sourceMappingURL=DropdownDivider.js.map
