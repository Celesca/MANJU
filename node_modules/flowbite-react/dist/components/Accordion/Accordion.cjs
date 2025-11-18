'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var chevronDownIcon = require('../../icons/chevron-down-icon.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

function Accordion(props) {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.accordionTheme.root, provider$1.theme?.accordion?.root, props.theme],
    [get.get(provider$1.clearTheme, "accordion.root"), props.clearTheme],
    [get.get(provider$1.applyTheme, "accordion.root"), props.applyTheme]
  );
  const {
    alwaysOpen = false,
    arrowIcon = chevronDownIcon.ChevronDownIcon,
    children,
    flush = false,
    collapseAll = false,
    className,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.accordion);
  const [isOpen, setOpen] = React.useState(collapseAll ? -1 : 0);
  const panels = React.useMemo(
    () => React.Children.map(
      children,
      (child, i) => React.cloneElement(child, {
        alwaysOpen,
        arrowIcon,
        flush,
        isOpen: isOpen === i,
        setOpen: () => setOpen(isOpen === i ? -1 : i)
      })
    ),
    [alwaysOpen, arrowIcon, children, flush, isOpen]
  );
  return /* @__PURE__ */ jsxRuntime.jsx(
    "div",
    {
      className: tailwindMerge.twMerge(theme$1.base, theme$1.flush[flush ? "on" : "off"], className),
      "data-testid": "flowbite-accordion",
      ...restProps,
      children: panels
    }
  );
}
Accordion.displayName = "Accordion";

exports.Accordion = Accordion;
//# sourceMappingURL=Accordion.cjs.map
