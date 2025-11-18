'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { hrTheme } from './theme.js';

const HRText = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [hrTheme.text, provider.theme?.hr?.text, props.theme],
    [get(provider.clearTheme, "hr.text"), props.clearTheme],
    [get(provider.applyTheme, "hr.text"), props.applyTheme]
  );
  const { className, text, ...restProps } = resolveProps(props, provider.props?.hrText);
  return /* @__PURE__ */ jsxs("div", { className: theme.base, children: [
    /* @__PURE__ */ jsx(
      "hr",
      {
        ref,
        className: twMerge(theme.hrLine, className),
        "data-testid": "flowbite-hr-text",
        role: "separator",
        ...restProps
      }
    ),
    /* @__PURE__ */ jsx("span", { className: theme.text, children: text })
  ] });
});
HRText.displayName = "HRText";

export { HRText };
//# sourceMappingURL=HRText.js.map
