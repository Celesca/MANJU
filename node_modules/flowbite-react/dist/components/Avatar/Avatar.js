'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { avatarTheme } from './theme.js';

const Avatar = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [avatarTheme, provider.theme?.avatar, props.theme],
    [get(provider.clearTheme, "avatar"), props.clearTheme],
    [get(provider.applyTheme, "avatar"), props.applyTheme]
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
  } = resolveProps(props, provider.props?.avatar);
  const imgClassName = twMerge(
    theme.root.img.base,
    bordered && theme.root.bordered,
    bordered && theme.root.color[color],
    rounded && theme.root.rounded,
    stacked && theme.root.stacked,
    theme.root.img.on,
    theme.root.size[size]
  );
  const imgProps = {
    className: twMerge(imgClassName, theme.root.img.on),
    "data-testid": "flowbite-avatar-img"
  };
  return /* @__PURE__ */ jsxs("div", { ref, className: twMerge(theme.root.base, className), "data-testid": "flowbite-avatar", ...restProps, children: [
    /* @__PURE__ */ jsxs("div", { className: theme.root.inner, children: [
      img ? typeof img === "string" ? /* @__PURE__ */ jsx("img", { alt, src: img, ...imgProps }) : img({ alt, ...imgProps }) : placeholderInitials ? /* @__PURE__ */ jsx(
        "div",
        {
          className: twMerge(
            theme.root.img.off,
            theme.root.initials.base,
            stacked && theme.root.stacked,
            bordered && theme.root.bordered,
            bordered && theme.root.color[color],
            theme.root.size[size],
            rounded && theme.root.rounded
          ),
          "data-testid": "flowbite-avatar-initials-placeholder",
          children: /* @__PURE__ */ jsx("span", { className: twMerge(theme.root.initials.text), "data-testid": "flowbite-avatar-initials-placeholder-text", children: placeholderInitials })
        }
      ) : /* @__PURE__ */ jsx("div", { className: twMerge(imgClassName, theme.root.img.off), "data-testid": "flowbite-avatar-img", children: /* @__PURE__ */ jsx(
        "svg",
        {
          className: theme.root.img.placeholder,
          fill: "currentColor",
          viewBox: "0 0 20 20",
          xmlns: "http://www.w3.org/2000/svg",
          children: /* @__PURE__ */ jsx("path", { fillRule: "evenodd", d: "M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z", clipRule: "evenodd" })
        }
      ) }),
      status && /* @__PURE__ */ jsx(
        "span",
        {
          "data-testid": "flowbite-avatar-status",
          className: twMerge(
            theme.root.status.base,
            theme.root.status[status],
            theme.root.statusPosition[statusPosition]
          )
        }
      )
    ] }),
    children && /* @__PURE__ */ jsx("div", { children })
  ] });
});
Avatar.displayName = "Avatar";

export { Avatar };
//# sourceMappingURL=Avatar.js.map
