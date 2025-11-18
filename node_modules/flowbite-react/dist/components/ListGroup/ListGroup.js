'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { listGroupTheme } from './theme.js';

const ListGroup = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [listGroupTheme.root, provider.theme?.listGroup?.root, props.theme],
    [get(provider.clearTheme, "listGroup.root"), props.clearTheme],
    [get(provider.applyTheme, "listGroup.root"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.listGroup);
  return /* @__PURE__ */ jsx("ul", { ref, className: twMerge(theme.base, className), ...restProps });
});
ListGroup.displayName = "ListGroup";

export { ListGroup };
//# sourceMappingURL=ListGroup.js.map
