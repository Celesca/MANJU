'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterTitle = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.title, provider.theme?.footer?.title, props.theme],
    [get(provider.clearTheme, "footer.title"), props.clearTheme],
    [get(provider.applyTheme, "footer.title"), props.applyTheme]
  );
  const { as: Component = "h2", className, title, ...restProps } = resolveProps(props, provider.props?.footerTitle);
  return /* @__PURE__ */ jsx(Component, { ref, "data-testid": "flowbite-footer-title", className: twMerge(theme.base, className), ...restProps, children: title });
});
FooterTitle.displayName = "FooterTitle";

export { FooterTitle };
//# sourceMappingURL=FooterTitle.js.map
