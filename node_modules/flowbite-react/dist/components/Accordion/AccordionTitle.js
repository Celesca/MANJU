'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useAccordionContext } from './AccordionPanelContext.js';
import { accordionTheme } from './theme.js';

function AccordionTitle(props) {
  const { arrowIcon: ArrowIcon, flush, isOpen, setOpen } = useAccordionContext();
  const onClick = () => typeof setOpen !== "undefined" && setOpen();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [accordionTheme.title, provider.theme?.accordion?.title, props.theme],
    [get(provider.clearTheme, "accordion.title"), props.clearTheme],
    [get(provider.applyTheme, "accordion.title"), props.applyTheme]
  );
  const { as: Heading = "h2", children, className, ...restProps } = resolveProps(props, provider.props?.accordionTitle);
  return /* @__PURE__ */ jsxs(
    "button",
    {
      className: twMerge(theme.base, theme.flush[flush ? "on" : "off"], theme.open[isOpen ? "on" : "off"], className),
      onClick,
      type: "button",
      ...restProps,
      children: [
        /* @__PURE__ */ jsx(Heading, { className: theme.heading, "data-testid": "flowbite-accordion-heading", children }),
        ArrowIcon && /* @__PURE__ */ jsx(
          ArrowIcon,
          {
            "aria-hidden": true,
            className: twMerge(theme.arrow.base, theme.arrow.open[isOpen ? "on" : "off"]),
            "data-testid": "flowbite-accordion-arrow"
          }
        )
      ]
    }
  );
}
AccordionTitle.displayName = "AccordionTitle";

export { AccordionTitle };
//# sourceMappingURL=AccordionTitle.js.map
