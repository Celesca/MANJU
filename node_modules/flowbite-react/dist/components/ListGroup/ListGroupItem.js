'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { listGroupTheme } from './theme.js';

const ListGroupItem = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [listGroupTheme.item, provider.theme?.listGroup?.item, props.theme],
    [get(provider.clearTheme, "listGroup.item"), props.clearTheme],
    [get(provider.applyTheme, "listGroup.item"), props.applyTheme]
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
  } = resolveProps(props, provider.props?.listGroupItem);
  const isLink = typeof href !== "undefined";
  const Component = isLink ? "a" : "button";
  return /* @__PURE__ */ jsx("li", { ref, className: twMerge(theme.base, className), children: /* @__PURE__ */ jsxs(
    Component,
    {
      href,
      onClick,
      type: isLink ? void 0 : "button",
      disabled,
      className: twMerge(
        theme.link.active[isActive ? "on" : "off"],
        theme.link.disabled[disabled ? "on" : "off"],
        theme.link.base,
        theme.link.href[isLink ? "on" : "off"]
      ),
      ...restProps,
      children: [
        Icon && /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, "data-testid": "flowbite-list-group-item-icon", className: theme.link.icon }),
        children
      ]
    }
  ) });
});
ListGroupItem.displayName = "ListGroupItem";

export { ListGroupItem };
//# sourceMappingURL=ListGroupItem.js.map
