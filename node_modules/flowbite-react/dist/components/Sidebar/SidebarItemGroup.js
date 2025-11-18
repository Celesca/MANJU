'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useSidebarContext } from './SidebarContext.js';
import { SidebarItemContext } from './SidebarItemContext.js';
import { sidebarTheme } from './theme.js';

const SidebarItemGroup = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useSidebarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme.itemGroup, provider.theme?.sidebar?.itemGroup, rootTheme?.itemGroup, props.theme],
    [get(provider.clearTheme, "sidebar.itemGroup"), get(rootClearTheme, "itemGroup"), props.clearTheme],
    [get(provider.applyTheme, "sidebar.itemGroup"), get(rootApplyTheme, "itemGroup"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.sidebarItemGroup);
  return /* @__PURE__ */ jsx(SidebarItemContext.Provider, { value: { isInsideCollapse: false }, children: /* @__PURE__ */ jsx(
    "ul",
    {
      ref,
      "data-testid": "flowbite-sidebar-item-group",
      className: twMerge(theme.base, className),
      ...restProps
    }
  ) });
});
SidebarItemGroup.displayName = "SidebarItemGroup";

export { SidebarItemGroup };
//# sourceMappingURL=SidebarItemGroup.js.map
