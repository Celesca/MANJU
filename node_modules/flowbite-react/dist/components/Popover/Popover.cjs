'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var react = require('@floating-ui/react');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var useFloating = require('../../hooks/use-floating.cjs');
var provider = require('../../theme/provider.cjs');
var helpers = require('../Floating/helpers.cjs');
var theme = require('./theme.cjs');

function Popover(props) {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.popoverTheme, provider$1.theme?.popover, props.theme],
    [get.get(provider$1.clearTheme, "popover"), props.clearTheme],
    [get.get(provider$1.applyTheme, "popover"), props.applyTheme]
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
  } = resolveProps.resolveProps(props, provider$1.props?.popover);
  const [uncontrolledOpen, setUncontrolledOpen] = React.useState(Boolean(initialOpen));
  const arrowRef = React.useRef(null);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = setControlledOpen ?? setUncontrolledOpen;
  const floatingProps = useFloating.useBaseFloating({
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
  const { getFloatingProps, getReferenceProps } = useFloating.useFloatingInteractions({
    context,
    role: "dialog",
    trigger
  });
  const childrenRef = children.ref;
  const ref = react.useMergeRefs([context.refs.setReference, childrenRef]);
  if (!React.isValidElement(children)) {
    throw Error("Invalid target element");
  }
  const target = React.useMemo(() => {
    return React.cloneElement(
      children,
      getReferenceProps({
        ref,
        "data-testid": "flowbite-popover-target",
        ...children?.props
      })
    );
  }, [children, ref, getReferenceProps]);
  return /* @__PURE__ */ jsxRuntime.jsxs(jsxRuntime.Fragment, { children: [
    target,
    open && /* @__PURE__ */ jsxRuntime.jsx(react.FloatingFocusManager, { context, modal: true, children: /* @__PURE__ */ jsxRuntime.jsx(
      "div",
      {
        className: theme$1.base,
        ref: refs.setFloating,
        "data-testid": "flowbite-popover",
        ...restProps,
        style: floatingStyles,
        ...getFloatingProps(),
        children: /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.inner, children: [
          arrow && /* @__PURE__ */ jsxRuntime.jsx(
            "div",
            {
              className: theme$1.arrow.base,
              "data-testid": "flowbite-popover-arrow",
              ref: arrowRef,
              style: {
                top: arrowY ?? " ",
                left: arrowX ?? " ",
                right: " ",
                bottom: " ",
                [helpers.getArrowPlacement({ placement })]: theme$1.arrow.placement
              },
              children: "\xA0"
            }
          ),
          /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.content, children: content })
        ] })
      }
    ) })
  ] });
}
Popover.displayName = "Popover";

exports.Popover = Popover;
//# sourceMappingURL=Popover.cjs.map
