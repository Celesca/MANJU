'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const Breadcrumb = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.breadcrumbTheme.root, provider$1.theme?.breadcrumb?.root, props.theme],
    [get.get(provider$1.clearTheme, "breadcrumb.root"), props.clearTheme],
    [get.get(provider$1.applyTheme, "breadcrumb.root"), props.applyTheme]
  );
  const { children, className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.breadcrumb);
  return /* @__PURE__ */ jsxRuntime.jsx("nav", { ref, "aria-label": "Breadcrumb", className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: /* @__PURE__ */ jsxRuntime.jsx("ol", { className: theme$1.list, children }) });
});
Breadcrumb.displayName = "Breadcrumb";

exports.Breadcrumb = Breadcrumb;
//# sourceMappingURL=Breadcrumb.cjs.map
