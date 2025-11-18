'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var xIcon = require('../../icons/x-icon.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const Alert = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.alertTheme, provider$1.theme?.alert, props.theme],
    [get.get(provider$1.clearTheme, "alert"), props.clearTheme],
    [get.get(provider$1.applyTheme, "alert"), props.applyTheme]
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
  } = resolveProps.resolveProps(props, provider$1.props?.alert);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "div",
    {
      ref,
      className: tailwindMerge.twMerge(
        theme$1.base,
        theme$1.color[color],
        rounded && theme$1.rounded,
        withBorderAccent && theme$1.borderAccent,
        className
      ),
      role: "alert",
      ...restProps,
      children: [
        /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.wrapper, "data-testid": "flowbite-alert-wrapper", children: [
          Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.icon, "data-testid": "flowbite-alert-icon" }),
          /* @__PURE__ */ jsxRuntime.jsx("div", { children }),
          typeof onDismiss === "function" && /* @__PURE__ */ jsxRuntime.jsx(
            "button",
            {
              "aria-label": "Dismiss",
              className: tailwindMerge.twMerge(theme$1.closeButton.base, theme$1.closeButton.color[color]),
              onClick: onDismiss,
              type: "button",
              children: /* @__PURE__ */ jsxRuntime.jsx(xIcon.XIcon, { "aria-hidden": true, className: theme$1.closeButton.icon })
            }
          )
        ] }),
        additionalContent && /* @__PURE__ */ jsxRuntime.jsx("div", { children: additionalContent })
      ]
    }
  );
});
Alert.displayName = "Alert";

exports.Alert = Alert;
//# sourceMappingURL=Alert.cjs.map
