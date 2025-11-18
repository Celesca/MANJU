'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useDrawerContext } from './DrawerContext.js';
import { drawerTheme } from './theme.js';

const DrawerItems = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useDrawerContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [drawerTheme.items, provider.theme?.drawer?.items, rootTheme?.items, props.theme],
    [get(provider.clearTheme, "drawer.items"), get(rootClearTheme, "items"), props.clearTheme],
    [get(provider.applyTheme, "drawer.items"), get(rootApplyTheme, "items"), props.applyTheme]
  );
  const { children, className, ...restProps } = resolveProps(props, provider.props?.drawerItems);
  return /* @__PURE__ */ jsx("div", { ref, "data-testid": "flowbite-drawer-items", className: twMerge(theme.base, className), ...restProps, children });
});
DrawerItems.displayName = "DrawerItems";

export { DrawerItems };
//# sourceMappingURL=DrawerItems.js.map
