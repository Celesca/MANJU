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

const ListItem = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.listTheme.item, provider$1.theme?.list?.item, props.theme],
    [get.get(provider$1.clearTheme, "list.item"), props.clearTheme],
    [get.get(provider$1.applyTheme, "list.item"), props.applyTheme]
  );
  const { children, className, icon: Icon, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.listItem);
  return /* @__PURE__ */ jsxRuntime.jsxs("li", { ref, className: tailwindMerge.twMerge(theme$1.withIcon[Icon ? "on" : "off"], className), ...restProps, children: [
    Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: tailwindMerge.twMerge(theme$1.icon) }),
    children
  ] });
});
ListItem.displayName = "ListItem";

exports.ListItem = ListItem;
//# sourceMappingURL=ListItem.cjs.map
