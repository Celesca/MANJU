'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { ChevronRightIcon } from '../../icons/chevron-right-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { breadcrumbTheme } from './theme.js';

const BreadcrumbItem = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [breadcrumbTheme.item, provider.theme?.breadcrumb?.item, props.theme],
    [get(provider.clearTheme, "breadcrumb.item"), props.clearTheme],
    [get(provider.applyTheme, "breadcrumb.item"), props.applyTheme]
  );
  const { children, className, href, icon: Icon, ...restProps } = resolveProps(props, provider.props?.breadcrumbItem);
  const isLink = typeof href !== "undefined";
  const Component = isLink ? "a" : "span";
  return /* @__PURE__ */ jsxs("li", { className: twMerge(theme.base, className), ...restProps, children: [
    /* @__PURE__ */ jsx(ChevronRightIcon, { "aria-hidden": true, className: theme.chevron, "data-testid": "flowbite-breadcrumb-separator" }),
    /* @__PURE__ */ jsxs(
      Component,
      {
        ref,
        className: theme.href[isLink ? "on" : "off"],
        "data-testid": "flowbite-breadcrumb-item",
        href,
        children: [
          Icon && /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, className: theme.icon }),
          children
        ]
      }
    )
  ] });
});
BreadcrumbItem.displayName = "BreadcrumbItem";

export { BreadcrumbItem };
//# sourceMappingURL=BreadcrumbItem.js.map
