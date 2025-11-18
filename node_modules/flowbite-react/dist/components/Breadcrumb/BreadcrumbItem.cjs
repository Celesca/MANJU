'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var chevronRightIcon = require('../../icons/chevron-right-icon.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const BreadcrumbItem = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.breadcrumbTheme.item, provider$1.theme?.breadcrumb?.item, props.theme],
    [get.get(provider$1.clearTheme, "breadcrumb.item"), props.clearTheme],
    [get.get(provider$1.applyTheme, "breadcrumb.item"), props.applyTheme]
  );
  const { children, className, href, icon: Icon, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.breadcrumbItem);
  const isLink = typeof href !== "undefined";
  const Component = isLink ? "a" : "span";
  return /* @__PURE__ */ jsxRuntime.jsxs("li", { className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: [
    /* @__PURE__ */ jsxRuntime.jsx(chevronRightIcon.ChevronRightIcon, { "aria-hidden": true, className: theme$1.chevron, "data-testid": "flowbite-breadcrumb-separator" }),
    /* @__PURE__ */ jsxRuntime.jsxs(
      Component,
      {
        ref,
        className: theme$1.href[isLink ? "on" : "off"],
        "data-testid": "flowbite-breadcrumb-item",
        href,
        children: [
          Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, className: theme$1.icon }),
          children
        ]
      }
    )
  ] });
});
BreadcrumbItem.displayName = "BreadcrumbItem";

exports.BreadcrumbItem = BreadcrumbItem;
//# sourceMappingURL=BreadcrumbItem.cjs.map
