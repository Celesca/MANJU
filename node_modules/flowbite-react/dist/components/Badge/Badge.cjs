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

const Badge = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.badgeTheme, provider$1.theme?.badge, props.theme],
    [get.get(provider$1.clearTheme, "badge"), props.clearTheme],
    [get.get(provider$1.applyTheme, "badge"), props.applyTheme]
  );
  const {
    children,
    color = "info",
    icon: Icon,
    size = "xs",
    className,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.badge);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "span",
    {
      ref,
      className: tailwindMerge.twMerge(
        theme$1.root.base,
        theme$1.root.color[color],
        theme$1.root.size[size],
        theme$1.icon[Icon ? "on" : "off"],
        className
      ),
      "data-testid": "flowbite-badge",
      ...restProps,
      children: [
        Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, className: theme$1.icon.size[size], "data-testid": "flowbite-badge-icon" }),
        children && /* @__PURE__ */ jsxRuntime.jsx("span", { children })
      ]
    }
  );
});
Badge.displayName = "Badge";

exports.Badge = Badge;
//# sourceMappingURL=Badge.cjs.map
