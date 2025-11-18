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

const TextInput = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.textInputTheme, provider$1.theme?.textInput, props.theme],
    [get.get(provider$1.clearTheme, "textInput"), props.clearTheme],
    [get.get(provider$1.applyTheme, "textInput"), props.applyTheme]
  );
  const {
    addon,
    className,
    color = "gray",
    icon: Icon,
    rightIcon: RightIcon,
    shadow,
    sizing = "md",
    type = "text",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.textInput);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { className: tailwindMerge.twMerge(theme$1.base, className), children: [
    addon && /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.addon, children: addon }),
    /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.field.base, children: [
      Icon && /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.field.icon.base, children: /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.field.icon.svg }) }),
      RightIcon && /* @__PURE__ */ jsxRuntime.jsx("div", { "data-testid": "right-icon", className: theme$1.field.rightIcon.base, children: /* @__PURE__ */ jsxRuntime.jsx(RightIcon, { className: theme$1.field.rightIcon.svg }) }),
      /* @__PURE__ */ jsxRuntime.jsx(
        "input",
        {
          className: tailwindMerge.twMerge(
            theme$1.field.input.base,
            theme$1.field.input.colors[color],
            theme$1.field.input.sizes[sizing],
            theme$1.field.input.withIcon[Icon ? "on" : "off"],
            theme$1.field.input.withRightIcon[RightIcon ? "on" : "off"],
            theme$1.field.input.withAddon[addon ? "on" : "off"],
            theme$1.field.input.withShadow[shadow ? "on" : "off"]
          ),
          type,
          ...restProps,
          ref
        }
      )
    ] })
  ] });
});
TextInput.displayName = "TextInput";

exports.TextInput = TextInput;
//# sourceMappingURL=TextInput.cjs.map
