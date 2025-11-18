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

const Card = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.cardTheme, provider$1.theme?.card, props.theme],
    [get.get(provider$1.clearTheme, "card"), props.clearTheme],
    [get.get(provider$1.applyTheme, "card"), props.applyTheme]
  );
  const { children, className, horizontal, href, imgAlt, imgSrc, renderImage, ...restProps } = resolveProps.resolveProps(
    props,
    provider$1.props?.card
  );
  const Component = typeof href === "undefined" ? "div" : "a";
  return /* @__PURE__ */ jsxRuntime.jsxs(
    Component,
    {
      ref,
      "data-testid": "flowbite-card",
      href,
      className: tailwindMerge.twMerge(
        theme$1.root.base,
        theme$1.root.horizontal[horizontal ? "on" : "off"],
        href && theme$1.root.href,
        className
      ),
      ...restProps,
      children: [
        renderImage?.(theme$1, !!horizontal) ?? (imgSrc && /* @__PURE__ */ jsxRuntime.jsx(
          "img",
          {
            "data-testid": "flowbite-card-image",
            alt: imgAlt ?? "",
            src: imgSrc,
            className: tailwindMerge.twMerge(theme$1.img.base, theme$1.img.horizontal[props.horizontal ? "on" : "off"])
          }
        )),
        /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.root.children, children })
      ]
    }
  );
});
Card.displayName = "Card";

exports.Card = Card;
//# sourceMappingURL=Card.cjs.map
