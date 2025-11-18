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
var ToastContext = require('./ToastContext.cjs');

const ToastToggle = React.forwardRef((props, ref) => {
  const {
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme,
    duration,
    isClosed,
    isRemoved,
    setIsClosed,
    setIsRemoved
  } = ToastContext.useToastContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.toastTheme.toggle, provider$1.theme?.toast?.toggle, rootTheme?.toggle, props.theme],
    [get.get(provider$1.clearTheme, "toast.toggle"), get.get(rootClearTheme, "toggle"), props.clearTheme],
    [get.get(provider$1.applyTheme, "toast.toggle"), get.get(rootApplyTheme, "toggle"), props.applyTheme]
  );
  const {
    className,
    onClick,
    onDismiss,
    xIcon: XIcon = xIcon.XIcon,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.toastToggle);
  function handleClick(e) {
    if (onClick) {
      onClick(e);
    }
    if (onDismiss) {
      onDismiss();
      return;
    }
    setIsClosed(!isClosed);
    setTimeout(() => setIsRemoved(!isRemoved), duration);
  }
  return /* @__PURE__ */ jsxRuntime.jsx(
    "button",
    {
      ref,
      "aria-label": "Close",
      onClick: handleClick,
      type: "button",
      className: tailwindMerge.twMerge(theme$1.base, className),
      ...restProps,
      children: /* @__PURE__ */ jsxRuntime.jsx(XIcon, { "aria-hidden": true, className: theme$1.icon })
    }
  );
});
ToastToggle.displayName = "ToastToggle";

exports.ToastToggle = ToastToggle;
//# sourceMappingURL=ToastToggle.cjs.map
