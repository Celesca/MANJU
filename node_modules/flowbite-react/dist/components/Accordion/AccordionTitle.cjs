'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var AccordionPanelContext = require('./AccordionPanelContext.cjs');
var theme = require('./theme.cjs');

function AccordionTitle(props) {
  const { arrowIcon: ArrowIcon, flush, isOpen, setOpen } = AccordionPanelContext.useAccordionContext();
  const onClick = () => typeof setOpen !== "undefined" && setOpen();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.accordionTheme.title, provider$1.theme?.accordion?.title, props.theme],
    [get.get(provider$1.clearTheme, "accordion.title"), props.clearTheme],
    [get.get(provider$1.applyTheme, "accordion.title"), props.applyTheme]
  );
  const { as: Heading = "h2", children, className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.accordionTitle);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "button",
    {
      className: tailwindMerge.twMerge(theme$1.base, theme$1.flush[flush ? "on" : "off"], theme$1.open[isOpen ? "on" : "off"], className),
      onClick,
      type: "button",
      ...restProps,
      children: [
        /* @__PURE__ */ jsxRuntime.jsx(Heading, { className: theme$1.heading, "data-testid": "flowbite-accordion-heading", children }),
        ArrowIcon && /* @__PURE__ */ jsxRuntime.jsx(
          ArrowIcon,
          {
            "aria-hidden": true,
            className: tailwindMerge.twMerge(theme$1.arrow.base, theme$1.arrow.open[isOpen ? "on" : "off"]),
            "data-testid": "flowbite-accordion-arrow"
          }
        )
      ]
    }
  );
}
AccordionTitle.displayName = "AccordionTitle";

exports.AccordionTitle = AccordionTitle;
//# sourceMappingURL=AccordionTitle.cjs.map
