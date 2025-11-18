'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { XIcon } from '../../icons/x-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { alertTheme } from './theme.js';

const Alert = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [alertTheme, provider.theme?.alert, props.theme],
    [get(provider.clearTheme, "alert"), props.clearTheme],
    [get(provider.applyTheme, "alert"), props.applyTheme]
  );
  const {
    additionalContent,
    children,
    className,
    color = "info",
    icon: Icon,
    onDismiss,
    rounded = true,
    withBorderAccent,
    ...restProps
  } = resolveProps(props, provider.props?.alert);
  return /* @__PURE__ */ jsxs(
    "div",
    {
      ref,
      className: twMerge(
        theme.base,
        theme.color[color],
        rounded && theme.rounded,
        withBorderAccent && theme.borderAccent,
        className
      ),
      role: "alert",
      ...restProps,
      children: [
        /* @__PURE__ */ jsxs("div", { className: theme.wrapper, "data-testid": "flowbite-alert-wrapper", children: [
          Icon && /* @__PURE__ */ jsx(Icon, { className: theme.icon, "data-testid": "flowbite-alert-icon" }),
          /* @__PURE__ */ jsx("div", { children }),
          typeof onDismiss === "function" && /* @__PURE__ */ jsx(
            "button",
            {
              "aria-label": "Dismiss",
              className: twMerge(theme.closeButton.base, theme.closeButton.color[color]),
              onClick: onDismiss,
              type: "button",
              children: /* @__PURE__ */ jsx(XIcon, { "aria-hidden": true, className: theme.closeButton.icon })
            }
          )
        ] }),
        additionalContent && /* @__PURE__ */ jsx("div", { children: additionalContent })
      ]
    }
  );
});
Alert.displayName = "Alert";

export { Alert };
//# sourceMappingURL=Alert.js.map
