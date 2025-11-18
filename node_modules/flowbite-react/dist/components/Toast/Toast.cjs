'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');
var ToastContext = require('./ToastContext.cjs');

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
const Toast = React.forwardRef((props, ref) => {
  const [isClosed, setIsClosed] = React.useState(false);
  const [isRemoved, setIsRemoved] = React.useState(false);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.toastTheme, provider$1.theme?.toast, props.theme],
    [get.get(provider$1.clearTheme, "toast"), props.clearTheme],
    [get.get(provider$1.applyTheme, "toast"), props.applyTheme]
  );
  const { className, duration = 300, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.toast);
  if (isRemoved) {
    return null;
  }
  return /* @__PURE__ */ jsxRuntime.jsx(
    ToastContext.ToastContext.Provider,
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
      children: /* @__PURE__ */ jsxRuntime.jsx(
        "div",
        {
          ref,
          "data-testid": "flowbite-toast",
          role: "alert",
          className: tailwindMerge.twMerge(theme$1.root.base, durationClasses[duration], isClosed && theme$1.root.closed, className),
          ...restProps
        }
      )
    }
  );
});
Toast.displayName = "Toast";

exports.Toast = Toast;
//# sourceMappingURL=Toast.cjs.map
