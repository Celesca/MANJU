'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { QuoteRightIcon } from '../../icons/quote-right-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { hrTheme } from './theme.js';

const HRIcon = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [hrTheme.icon, provider.theme?.hr?.icon, props.theme],
    [get(provider.clearTheme, "hr.icon"), props.clearTheme],
    [get(provider.applyTheme, "hr.icon"), props.applyTheme]
  );
  const { icon: Icon = QuoteRightIcon, className, ...restProps } = resolveProps(props, provider.props?.hrIcon);
  return /* @__PURE__ */ jsxs("div", { className: theme.base, children: [
    /* @__PURE__ */ jsx(
      "hr",
      {
        ref,
        className: twMerge(theme.hrLine, className),
        "data-testid": "flowbite-hr-icon",
        role: "separator",
        ...restProps
      }
    ),
    /* @__PURE__ */ jsx("div", { className: theme.icon.base, children: /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, className: theme.icon.icon }) })
  ] });
});
HRIcon.displayName = "HRIcon";

export { HRIcon };
//# sourceMappingURL=HRIcon.js.map
