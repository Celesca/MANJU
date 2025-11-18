'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { listTheme } from './theme.js';

const List = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [listTheme.root, provider.theme?.list?.root, props.theme],
    [get(provider.clearTheme, "list.root"), props.clearTheme],
    [get(provider.applyTheme, "list.root"), props.applyTheme]
  );
  const { className, horizontal, nested, ordered, unstyled, ...restProps } = resolveProps(props, provider.props?.list);
  const Component = ordered ? "ol" : "ul";
  return /* @__PURE__ */ jsx(
    Component,
    {
      ref,
      className: twMerge(
        theme.base,
        theme.ordered[ordered ? "on" : "off"],
        unstyled && theme.unstyled,
        nested && theme.nested,
        horizontal && theme.horizontal,
        className
      ),
      ...restProps
    }
  );
});
List.displayName = "List";

export { List };
//# sourceMappingURL=List.js.map
