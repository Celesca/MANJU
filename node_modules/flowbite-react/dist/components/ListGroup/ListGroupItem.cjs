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

const ListGroupItem = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.listGroupTheme.item, provider$1.theme?.listGroup?.item, props.theme],
    [get.get(provider$1.clearTheme, "listGroup.item"), props.clearTheme],
    [get.get(provider$1.applyTheme, "listGroup.item"), props.applyTheme]
  );
  const {
    active: isActive,
    children,
    className,
    href,
    icon: Icon,
    onClick,
    disabled,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.listGroupItem);
  const isLink = typeof href !== "undefined";
  const Component = isLink ? "a" : "button";
  return /* @__PURE__ */ jsxRuntime.jsx("li", { ref, className: tailwindMerge.twMerge(theme$1.base, className), children: /* @__PURE__ */ jsxRuntime.jsxs(
    Component,
    {
      href,
      onClick,
      type: isLink ? void 0 : "button",
      disabled,
      className: tailwindMerge.twMerge(
        theme$1.link.active[isActive ? "on" : "off"],
        theme$1.link.disabled[disabled ? "on" : "off"],
        theme$1.link.base,
        theme$1.link.href[isLink ? "on" : "off"]
      ),
      ...restProps,
      children: [
        Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, "data-testid": "flowbite-list-group-item-icon", className: theme$1.link.icon }),
        children
      ]
    }
  ) });
});
ListGroupItem.displayName = "ListGroupItem";

exports.ListGroupItem = ListGroupItem;
//# sourceMappingURL=ListGroupItem.cjs.map
