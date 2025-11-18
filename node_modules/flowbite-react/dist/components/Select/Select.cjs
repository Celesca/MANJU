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

const Select = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.selectTheme, provider$1.theme?.select, props.theme],
    [get.get(provider$1.clearTheme, "select"), props.clearTheme],
    [get.get(provider$1.applyTheme, "select"), props.applyTheme]
  );
  const {
    addon,
    className,
    color = "gray",
    icon: Icon,
    shadow,
    sizing = "md",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.select);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { className: tailwindMerge.twMerge(theme$1.base, className), children: [
    addon && /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.addon, children: addon }),
    /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.field.base, children: [
      Icon && /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.field.icon.base, children: /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.field.icon.svg }) }),
      /* @__PURE__ */ jsxRuntime.jsx(
        "select",
        {
          ref,
          className: tailwindMerge.twMerge(
            theme$1.field.select.base,
            theme$1.field.select.colors[color],
            theme$1.field.select.sizes[sizing],
            theme$1.field.select.withIcon[Icon ? "on" : "off"],
            theme$1.field.select.withAddon[addon ? "on" : "off"],
            theme$1.field.select.withShadow[shadow ? "on" : "off"]
          ),
          ...restProps
        }
      )
    ] })
  ] });
});
Select.displayName = "Select";

exports.Select = Select;
//# sourceMappingURL=Select.cjs.map
