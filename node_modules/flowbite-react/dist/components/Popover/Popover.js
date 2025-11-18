'use client';
import { jsxs, Fragment, jsx } from 'react/jsx-runtime';
import { useMergeRefs, FloatingFocusManager } from '@floating-ui/react';
import { useState, useRef, isValidElement, useMemo, cloneElement } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { useBaseFloating, useFloatingInteractions } from '../../hooks/use-floating.js';
import { useThemeProvider } from '../../theme/provider.js';
import { getArrowPlacement } from '../Floating/helpers.js';
import { popoverTheme } from './theme.js';

function Popover(props) {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [popoverTheme, provider.theme?.popover, props.theme],
    [get(provider.clearTheme, "popover"), props.clearTheme],
    [get(provider.applyTheme, "popover"), props.applyTheme]
  );
  const {
    children,
    content,
    arrow = true,
    trigger = "click",
    initialOpen,
    open: controlledOpen,
    onOpenChange: setControlledOpen,
    placement: theirPlacement = "bottom",
    ...restProps
  } = resolveProps(props, provider.props?.popover);
  const [uncontrolledOpen, setUncontrolledOpen] = useState(Boolean(initialOpen));
  const arrowRef = useRef(null);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = setControlledOpen ?? setUncontrolledOpen;
  const floatingProps = useBaseFloating({
    open,
    placement: theirPlacement,
    arrowRef,
    setOpen
  });
  const {
    floatingStyles,
    context,
    placement,
    middlewareData: { arrow: { x: arrowX, y: arrowY } = {} },
    refs
  } = floatingProps;
  const { getFloatingProps, getReferenceProps } = useFloatingInteractions({
    context,
    role: "dialog",
    trigger
  });
  const childrenRef = children.ref;
  const ref = useMergeRefs([context.refs.setReference, childrenRef]);
  if (!isValidElement(children)) {
    throw Error("Invalid target element");
  }
  const target = useMemo(() => {
    return cloneElement(
      children,
      getReferenceProps({
        ref,
        "data-testid": "flowbite-popover-target",
        ...children?.props
      })
    );
  }, [children, ref, getReferenceProps]);
  return /* @__PURE__ */ jsxs(Fragment, { children: [
    target,
    open && /* @__PURE__ */ jsx(FloatingFocusManager, { context, modal: true, children: /* @__PURE__ */ jsx(
      "div",
      {
        className: theme.base,
        ref: refs.setFloating,
        "data-testid": "flowbite-popover",
        ...restProps,
        style: floatingStyles,
        ...getFloatingProps(),
        children: /* @__PURE__ */ jsxs("div", { className: theme.inner, children: [
          arrow && /* @__PURE__ */ jsx(
            "div",
            {
              className: theme.arrow.base,
              "data-testid": "flowbite-popover-arrow",
              ref: arrowRef,
              style: {
                top: arrowY ?? " ",
                left: arrowX ?? " ",
                right: " ",
                bottom: " ",
                [getArrowPlacement({ placement })]: theme.arrow.placement
              },
              children: "\xA0"
            }
          ),
          /* @__PURE__ */ jsx("div", { className: theme.content, children: content })
        ] })
      }
    ) })
  ] });
}
Popover.displayName = "Popover";

export { Popover };
//# sourceMappingURL=Popover.js.map
