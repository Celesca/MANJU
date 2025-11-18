'use client';
import { jsx } from 'react/jsx-runtime';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useAccordionContext } from './AccordionPanelContext.js';
import { accordionTheme } from './theme.js';

function AccordionContent(props) {
  const { isOpen } = useAccordionContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [accordionTheme.content, provider.theme?.accordion?.content, props.theme],
    [get(provider.clearTheme, "accordion.content"), props.clearTheme],
    [get(provider.applyTheme, "accordion.content"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.accordionContent);
  return /* @__PURE__ */ jsx(
    "div",
    {
      className: twMerge(theme.base, className),
      "data-testid": "flowbite-accordion-content",
      hidden: !isOpen,
      ...restProps
    }
  );
}
AccordionContent.displayName = "AccordionContent";

export { AccordionContent };
//# sourceMappingURL=AccordionContent.js.map
