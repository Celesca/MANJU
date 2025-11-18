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

const SidebarCTA = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, isCollapsed } = useSidebarContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [sidebarTheme.cta, provider.theme?.sidebar?.cta, rootTheme?.cta, props.theme],
    [get(provider.clearTheme, "sidebar.cta"), get(rootClearTheme, "cta"), props.clearTheme],
    [get(provider.applyTheme, "sidebar.cta"), get(rootApplyTheme, "cta"), props.applyTheme]
  );
  const { color = "info", className, ...restProps } = resolveProps(props, provider.props?.sidebarCTA);
  return /* @__PURE__ */ jsx(
    "div",
    {
      ref,
      "data-testid": "sidebar-cta",
      hidden: isCollapsed,
      className: twMerge(theme.base, theme.color[color], className),
      ...restProps
    }
  );
});
SidebarCTA.displayName = "SidebarCTA";

export { SidebarCTA };
//# sourceMappingURL=SidebarCTA.js.map
