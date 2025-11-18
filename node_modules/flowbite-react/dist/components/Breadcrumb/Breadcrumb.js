'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { breadcrumbTheme } from './theme.js';

const Breadcrumb = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [breadcrumbTheme.root, provider.theme?.breadcrumb?.root, props.theme],
    [get(provider.clearTheme, "breadcrumb.root"), props.clearTheme],
    [get(provider.applyTheme, "breadcrumb.root"), props.applyTheme]
  );
  const { children, className, ...restProps } = resolveProps(props, provider.props?.breadcrumb);
  return /* @__PURE__ */ jsx("nav", { ref, "aria-label": "Breadcrumb", className: twMerge(theme.base, className), ...restProps, children: /* @__PURE__ */ jsx("ol", { className: theme.list, children }) });
});
Breadcrumb.displayName = "Breadcrumb";

export { Breadcrumb };
//# sourceMappingURL=Breadcrumb.js.map
