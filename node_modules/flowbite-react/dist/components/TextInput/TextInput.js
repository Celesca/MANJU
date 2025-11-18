'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { textInputTheme } from './theme.js';

const TextInput = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [textInputTheme, provider.theme?.textInput, props.theme],
    [get(provider.clearTheme, "textInput"), props.clearTheme],
    [get(provider.applyTheme, "textInput"), props.applyTheme]
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
  } = resolveProps(props, provider.props?.textInput);
  return /* @__PURE__ */ jsxs("div", { className: twMerge(theme.base, className), children: [
    addon && /* @__PURE__ */ jsx("span", { className: theme.addon, children: addon }),
    /* @__PURE__ */ jsxs("div", { className: theme.field.base, children: [
      Icon && /* @__PURE__ */ jsx("div", { className: theme.field.icon.base, children: /* @__PURE__ */ jsx(Icon, { className: theme.field.icon.svg }) }),
      RightIcon && /* @__PURE__ */ jsx("div", { "data-testid": "right-icon", className: theme.field.rightIcon.base, children: /* @__PURE__ */ jsx(RightIcon, { className: theme.field.rightIcon.svg }) }),
      /* @__PURE__ */ jsx(
        "input",
        {
          className: twMerge(
            theme.field.input.base,
            theme.field.input.colors[color],
            theme.field.input.sizes[sizing],
            theme.field.input.withIcon[Icon ? "on" : "off"],
            theme.field.input.withRightIcon[RightIcon ? "on" : "off"],
            theme.field.input.withAddon[addon ? "on" : "off"],
            theme.field.input.withShadow[shadow ? "on" : "off"]
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

export { TextInput };
//# sourceMappingURL=TextInput.js.map
