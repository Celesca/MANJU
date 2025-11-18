'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { selectTheme } from './theme.js';

const Select = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [selectTheme, provider.theme?.select, props.theme],
    [get(provider.clearTheme, "select"), props.clearTheme],
    [get(provider.applyTheme, "select"), props.applyTheme]
  );
  const {
    addon,
    className,
    color = "gray",
    icon: Icon,
    shadow,
    sizing = "md",
    ...restProps
  } = resolveProps(props, provider.props?.select);
  return /* @__PURE__ */ jsxs("div", { className: twMerge(theme.base, className), children: [
    addon && /* @__PURE__ */ jsx("span", { className: theme.addon, children: addon }),
    /* @__PURE__ */ jsxs("div", { className: theme.field.base, children: [
      Icon && /* @__PURE__ */ jsx("div", { className: theme.field.icon.base, children: /* @__PURE__ */ jsx(Icon, { className: theme.field.icon.svg }) }),
      /* @__PURE__ */ jsx(
        "select",
        {
          ref,
          className: twMerge(
            theme.field.select.base,
            theme.field.select.colors[color],
            theme.field.select.sizes[sizing],
            theme.field.select.withIcon[Icon ? "on" : "off"],
            theme.field.select.withAddon[addon ? "on" : "off"],
            theme.field.select.withShadow[shadow ? "on" : "off"]
          ),
          ...restProps
        }
      )
    ] })
  ] });
});
Select.displayName = "Select";

export { Select };
//# sourceMappingURL=Select.js.map
