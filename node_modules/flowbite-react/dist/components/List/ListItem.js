'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { listTheme } from './theme.js';

const ListItem = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [listTheme.item, provider.theme?.list?.item, props.theme],
    [get(provider.clearTheme, "list.item"), props.clearTheme],
    [get(provider.applyTheme, "list.item"), props.applyTheme]
  );
  const { children, className, icon: Icon, ...restProps } = resolveProps(props, provider.props?.listItem);
  return /* @__PURE__ */ jsxs("li", { ref, className: twMerge(theme.withIcon[Icon ? "on" : "off"], className), ...restProps, children: [
    Icon && /* @__PURE__ */ jsx(Icon, { className: twMerge(theme.icon) }),
    children
  ] });
});
ListItem.displayName = "ListItem";

export { ListItem };
//# sourceMappingURL=ListItem.js.map
