'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { blockquoteTheme } from './theme.js';

const Blockquote = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [blockquoteTheme, provider.theme?.blockquote, props.theme],
    [get(provider.clearTheme, "blockquote"), props.clearTheme],
    [get(provider.applyTheme, "blockquote"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.blockquote);
  return /* @__PURE__ */ jsx(
    "blockquote",
    {
      ref,
      className: twMerge(theme.root.base, className),
      "data-testid": "flowbite-blockquote",
      ...restProps
    }
  );
});
Blockquote.displayName = "Blockquote";

export { Blockquote };
//# sourceMappingURL=Blockquote.js.map
