'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { XIcon } from '../../icons/x-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { toastTheme } from './theme.js';
import { useToastContext } from './ToastContext.js';

const ToastToggle = forwardRef((props, ref) => {
  const {
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme,
    duration,
    isClosed,
    isRemoved,
    setIsClosed,
    setIsRemoved
  } = useToastContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [toastTheme.toggle, provider.theme?.toast?.toggle, rootTheme?.toggle, props.theme],
    [get(provider.clearTheme, "toast.toggle"), get(rootClearTheme, "toggle"), props.clearTheme],
    [get(provider.applyTheme, "toast.toggle"), get(rootApplyTheme, "toggle"), props.applyTheme]
  );
  const {
    className,
    onClick,
    onDismiss,
    xIcon: XIcon$1 = XIcon,
    ...restProps
  } = resolveProps(props, provider.props?.toastToggle);
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
  return /* @__PURE__ */ jsx(
    "button",
    {
      ref,
      "aria-label": "Close",
      onClick: handleClick,
      type: "button",
      className: twMerge(theme.base, className),
      ...restProps,
      children: /* @__PURE__ */ jsx(XIcon$1, { "aria-hidden": true, className: theme.icon })
    }
  );
});
ToastToggle.displayName = "ToastToggle";

export { ToastToggle };
//# sourceMappingURL=ToastToggle.js.map
