'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterDivider = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.divider, provider.theme?.footer?.divider, props.theme],
    [get(provider.clearTheme, "footer.divider"), props.clearTheme],
    [get(provider.applyTheme, "footer.divider"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.footerDivider);
  return /* @__PURE__ */ jsx("hr", { ref, "data-testid": "footer-divider", className: twMerge(theme.base, className), ...restProps });
});
FooterDivider.displayName = "FooterDivider";

export { FooterDivider };
//# sourceMappingURL=FooterDivider.js.map
