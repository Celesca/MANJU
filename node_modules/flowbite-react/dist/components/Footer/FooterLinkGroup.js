'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterLinkGroup = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.groupLink, provider.theme?.footer?.groupLink, props.theme],
    [get(provider.clearTheme, "footer.groupLink"), props.clearTheme],
    [get(provider.applyTheme, "footer.groupLink"), props.applyTheme]
  );
  const { className, col = false, ...restProps } = resolveProps(props, provider.props?.footerLinkGroup);
  return /* @__PURE__ */ jsx(
    "ul",
    {
      ref,
      "data-testid": "footer-groupLink",
      className: twMerge(theme.base, col && theme.col, className),
      ...restProps
    }
  );
});
FooterLinkGroup.displayName = "FooterLinkGroup";

export { FooterLinkGroup };
//# sourceMappingURL=FooterLinkGroup.js.map
