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

const Avatar = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.avatarTheme, provider$1.theme?.avatar, props.theme],
    [get.get(provider$1.clearTheme, "avatar"), props.clearTheme],
    [get.get(provider$1.applyTheme, "avatar"), props.applyTheme]
  );
  const {
    alt = "",
    bordered = false,
    children,
    className,
    color = "light",
    img,
    placeholderInitials = "",
    rounded = false,
    size = "md",
    stacked = false,
    status,
    statusPosition = "top-left",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.avatar);
  const imgClassName = tailwindMerge.twMerge(
    theme$1.root.img.base,
    bordered && theme$1.root.bordered,
    bordered && theme$1.root.color[color],
    rounded && theme$1.root.rounded,
    stacked && theme$1.root.stacked,
    theme$1.root.img.on,
    theme$1.root.size[size]
  );
  const imgProps = {
    className: tailwindMerge.twMerge(imgClassName, theme$1.root.img.on),
    "data-testid": "flowbite-avatar-img"
  };
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { ref, className: tailwindMerge.twMerge(theme$1.root.base, className), "data-testid": "flowbite-avatar", ...restProps, children: [
    /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.root.inner, children: [
      img ? typeof img === "string" ? /* @__PURE__ */ jsxRuntime.jsx("img", { alt, src: img, ...imgProps }) : img({ alt, ...imgProps }) : placeholderInitials ? /* @__PURE__ */ jsxRuntime.jsx(
        "div",
        {
          className: tailwindMerge.twMerge(
            theme$1.root.img.off,
            theme$1.root.initials.base,
            stacked && theme$1.root.stacked,
            bordered && theme$1.root.bordered,
            bordered && theme$1.root.color[color],
            theme$1.root.size[size],
            rounded && theme$1.root.rounded
          ),
          "data-testid": "flowbite-avatar-initials-placeholder",
          children: /* @__PURE__ */ jsxRuntime.jsx("span", { className: tailwindMerge.twMerge(theme$1.root.initials.text), "data-testid": "flowbite-avatar-initials-placeholder-text", children: placeholderInitials })
        }
      ) : /* @__PURE__ */ jsxRuntime.jsx("div", { className: tailwindMerge.twMerge(imgClassName, theme$1.root.img.off), "data-testid": "flowbite-avatar-img", children: /* @__PURE__ */ jsxRuntime.jsx(
        "svg",
        {
          className: theme$1.root.img.placeholder,
          fill: "currentColor",
          viewBox: "0 0 20 20",
          xmlns: "http://www.w3.org/2000/svg",
          children: /* @__PURE__ */ jsxRuntime.jsx("path", { fillRule: "evenodd", d: "M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z", clipRule: "evenodd" })
        }
      ) }),
      status && /* @__PURE__ */ jsxRuntime.jsx(
        "span",
        {
          "data-testid": "flowbite-avatar-status",
          className: tailwindMerge.twMerge(
            theme$1.root.status.base,
            theme$1.root.status[status],
            theme$1.root.statusPosition[statusPosition]
          )
        }
      )
    ] }),
    children && /* @__PURE__ */ jsxRuntime.jsx("div", { children })
  ] });
});
Avatar.displayName = "Avatar";

exports.Avatar = Avatar;
//# sourceMappingURL=Avatar.cjs.map
