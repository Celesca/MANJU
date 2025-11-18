'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useSidebarContext } from './SidebarContext.js';
import { sidebarTheme } from './theme.js';

const SidebarItems = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useSidebarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme.items, provider.theme?.sidebar?.items, rootTheme?.items, props.theme],
    [get(provider.clearTheme, "sidebar.items"), get(rootClearTheme, "items"), props.clearTheme],
    [get(provider.applyTheme, "sidebar.items"), get(rootApplyTheme, "items"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.sidebarItems);
  return /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.base, className), "data-testid": "flowbite-sidebar-items", ...restProps });
});
SidebarItems.displayName = "SidebarItems";

export { SidebarItems };
//# sourceMappingURL=SidebarItems.js.map
