'use client';
import { jsx } from 'react/jsx-runtime';
import { useState, useMemo, Children, cloneElement } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { ChevronDownIcon } from '../../icons/chevron-down-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { accordionTheme } from './theme.js';

function Accordion(props) {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [accordionTheme.root, provider.theme?.accordion?.root, props.theme],
    [get(provider.clearTheme, "accordion.root"), props.clearTheme],
    [get(provider.applyTheme, "accordion.root"), props.applyTheme]
  );
  const {
    alwaysOpen = false,
    arrowIcon = ChevronDownIcon,
    children,
    flush = false,
    collapseAll = false,
    className,
    ...restProps
  } = resolveProps(props, provider.props?.accordion);
  const [isOpen, setOpen] = useState(collapseAll ? -1 : 0);
  const panels = useMemo(
    () => Children.map(
      children,
      (child, i) => cloneElement(child, {
        alwaysOpen,
        arrowIcon,
        flush,
        isOpen: isOpen === i,
        setOpen: () => setOpen(isOpen === i ? -1 : i)
      })
    ),
    [alwaysOpen, arrowIcon, children, flush, isOpen]
  );
  return /* @__PURE__ */ jsx(
    "div",
    {
      className: twMerge(theme.base, theme.flush[flush ? "on" : "off"], className),
      "data-testid": "flowbite-accordion",
      ...restProps,
      children: panels
    }
  );
}
Accordion.displayName = "Accordion";

export { Accordion };
//# sourceMappingURL=Accordion.js.map
