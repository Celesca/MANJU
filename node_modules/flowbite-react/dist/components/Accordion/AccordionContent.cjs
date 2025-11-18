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

function AccordionContent(props) {
  const { isOpen } = AccordionPanelContext.useAccordionContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.accordionTheme.content, provider$1.theme?.accordion?.content, props.theme],
    [get.get(provider$1.clearTheme, "accordion.content"), props.clearTheme],
    [get.get(provider$1.applyTheme, "accordion.content"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.accordionContent);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "div",
    {
      className: tailwindMerge.twMerge(theme$1.base, className),
      "data-testid": "flowbite-accordion-content",
      hidden: !isOpen,
      ...restProps
    }
  );
}
AccordionContent.displayName = "AccordionContent";

exports.AccordionContent = AccordionContent;
//# sourceMappingURL=AccordionContent.cjs.map
