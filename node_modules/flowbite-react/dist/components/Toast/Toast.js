'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef, useState } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { toastTheme } from './theme.js';
import { ToastContext } from './ToastContext.js';

const durationClasses = {
  75: "duration-75",
  100: "duration-100",
  150: "duration-150",
  200: "duration-200",
  300: "duration-300",
  500: "duration-500",
  700: "duration-700",
  1e3: "duration-1000"
};
const Toast = forwardRef((props, ref) => {
  const [isClosed, setIsClosed] = useState(false);
  const [isRemoved, setIsRemoved] = useState(false);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [toastTheme, provider.theme?.toast, props.theme],
    [get(provider.clearTheme, "toast"), props.clearTheme],
    [get(provider.applyTheme, "toast"), props.applyTheme]
  );
  const { className, duration = 300, ...restProps } = resolveProps(props, provider.props?.toast);
  if (isRemoved) {
    return null;
  }
  return /* @__PURE__ */ jsx(
    ToastContext.Provider,
    {
      value: {
        theme: props.theme,
        clearTheme: props.clearTheme,
        applyTheme: props.applyTheme,
        duration,
        isClosed,
        isRemoved,
        setIsClosed,
        setIsRemoved
      },
      children: /* @__PURE__ */ jsx(
        "div",
        {
          ref,
          "data-testid": "flowbite-toast",
          role: "alert",
          className: twMerge(theme.root.base, durationClasses[duration], isClosed && theme.root.closed, className),
          ...restProps
        }
      )
    }
  );
});
Toast.displayName = "Toast";

export { Toast };
//# sourceMappingURL=Toast.js.map
