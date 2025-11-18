'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { cardTheme } from './theme.js';

const Card = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [cardTheme, provider.theme?.card, props.theme],
    [get(provider.clearTheme, "card"), props.clearTheme],
    [get(provider.applyTheme, "card"), props.applyTheme]
  );
  const { children, className, horizontal, href, imgAlt, imgSrc, renderImage, ...restProps } = resolveProps(
    props,
    provider.props?.card
  );
  const Component = typeof href === "undefined" ? "div" : "a";
  return /* @__PURE__ */ jsxs(
    Component,
    {
      ref,
      "data-testid": "flowbite-card",
      href,
      className: twMerge(
        theme.root.base,
        theme.root.horizontal[horizontal ? "on" : "off"],
        href && theme.root.href,
        className
      ),
      ...restProps,
      children: [
        renderImage?.(theme, !!horizontal) ?? (imgSrc && /* @__PURE__ */ jsx(
          "img",
          {
            "data-testid": "flowbite-card-image",
            alt: imgAlt ?? "",
            src: imgSrc,
            className: twMerge(theme.img.base, theme.img.horizontal[props.horizontal ? "on" : "off"])
          }
        )),
        /* @__PURE__ */ jsx("div", { className: theme.root.children, children })
      ]
    }
  );
});
Card.displayName = "Card";

export { Card };
//# sourceMappingURL=Card.js.map
