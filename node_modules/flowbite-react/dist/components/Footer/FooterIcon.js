'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { footerTheme } from './theme.js';

const FooterIcon = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [footerTheme.icon, provider.theme?.footer?.icon, props.theme],
    [get(provider.clearTheme, "footer.icon"), props.clearTheme],
    [get(provider.applyTheme, "footer.icon"), props.applyTheme]
  );
  const { ariaLabel, className, href, icon: Icon, ...restProps } = resolveProps(props, provider.props?.footerIcon);
  return /* @__PURE__ */ jsx("div", { ref, children: href ? /* @__PURE__ */ jsx(
    "a",
    {
      "aria-label": ariaLabel,
      "data-testid": "flowbite-footer-icon",
      href,
      className: twMerge(theme.base, className),
      ...restProps,
      children: /* @__PURE__ */ jsx(Icon, { className: theme.size })
    }
  ) : /* @__PURE__ */ jsx(Icon, { "data-testid": "flowbite-footer-icon", className: theme.size, ...restProps }) });
});
FooterIcon.displayName = "FooterIcon";

export { FooterIcon };
//# sourceMappingURL=FooterIcon.js.map
