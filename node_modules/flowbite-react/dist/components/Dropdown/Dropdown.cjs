'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var react = require('@floating-ui/react');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var useFloating = require('../../hooks/use-floating.cjs');
var chevronDownIcon = require('../../icons/chevron-down-icon.cjs');
var chevronLeftIcon = require('../../icons/chevron-left-icon.cjs');
var chevronRightIcon = require('../../icons/chevron-right-icon.cjs');
var chevronUpIcon = require('../../icons/chevron-up-icon.cjs');
var provider = require('../../theme/provider.cjs');
var Button = require('../Button/Button.cjs');
require('../Button/ButtonGroup.cjs');
require('../Button/ButtonGroupContext.cjs');
var DropdownContext = require('./DropdownContext.cjs');
var theme = require('./theme.cjs');

const icons = {
  top: chevronUpIcon.ChevronUpIcon,
  right: chevronRightIcon.ChevronRightIcon,
  bottom: chevronDownIcon.ChevronDownIcon,
  left: chevronLeftIcon.ChevronLeftIcon
};
function Trigger({
  refs,
  children,
  inline,
  theme,
  disabled,
  setButtonWidth,
  getReferenceProps,
  renderTrigger,
  ...buttonProps
}) {
  const ref = refs.reference;
  const a11yProps = getReferenceProps();
  React.useEffect(() => {
    if (ref.current) {
      setButtonWidth?.(ref.current.clientWidth);
    }
  }, [ref, setButtonWidth]);
  if (renderTrigger) {
    const triggerElement = renderTrigger(theme);
    return React.cloneElement(triggerElement, { ref: refs.setReference, disabled, ...a11yProps, ...triggerElement.props });
  }
  return inline ? /* @__PURE__ */ jsxRuntime.jsx("button", { type: "button", ref: refs.setReference, className: theme?.inlineWrapper, disabled, ...a11yProps, children }) : /* @__PURE__ */ jsxRuntime.jsx(Button.Button, { ...buttonProps, disabled, type: "button", ref: refs.setReference, ...a11yProps, children });
}
function Dropdown(props) {
  const [open, setOpen] = React.useState(false);
  const [activeIndex, setActiveIndex] = React.useState(null);
  const [selectedIndex, setSelectedIndex] = React.useState(null);
  const [buttonWidth, setButtonWidth] = React.useState(void 0);
  const elementsRef = React.useRef([]);
  const labelsRef = React.useRef([]);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.dropdownTheme, provider$1.theme?.dropdown, props.theme],
    [get.get(provider$1.clearTheme, "dropdown"), props.clearTheme],
    [get.get(provider$1.applyTheme, "dropdown"), props.applyTheme]
  );
  const {
    children,
    className,
    dismissOnClick = true,
    enableTypeAhead = true,
    renderTrigger,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.dropdown);
  const {
    placement = restProps.inline ? "bottom-start" : "bottom",
    trigger = "click",
    label,
    inline,
    arrowIcon = true,
    ...buttonProps
  } = restProps;
  const dataTestId = restProps["data-testid"] || "flowbite-dropdown-target";
  const handleSelect = React.useCallback((index) => {
    setSelectedIndex(index);
    setOpen(false);
  }, []);
  const handleTypeaheadMatch = React.useCallback(
    (index) => {
      if (open) {
        setActiveIndex(index);
      } else {
        handleSelect(index);
      }
    },
    [open, handleSelect]
  );
  const { context, floatingStyles, refs } = useFloating.useBaseFloating({
    open,
    setOpen,
    placement
  });
  const listNav = react.useListNavigation(context, {
    listRef: elementsRef,
    activeIndex,
    selectedIndex,
    onNavigate: setActiveIndex
  });
  const typeahead = react.useTypeahead(context, {
    listRef: labelsRef,
    activeIndex,
    selectedIndex,
    onMatch: handleTypeaheadMatch,
    enabled: enableTypeAhead
  });
  const { getReferenceProps, getFloatingProps, getItemProps } = useFloating.useFloatingInteractions({
    context,
    role: "menu",
    trigger,
    interactions: [listNav, typeahead]
  });
  const Icon = React.useMemo(() => {
    const [p] = placement.split("-");
    return icons[p] ?? chevronDownIcon.ChevronDownIcon;
  }, [placement]);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    DropdownContext.DropdownContext.Provider,
    {
      value: {
        theme: props.theme,
        clearTheme: props.clearTheme,
        applyTheme: props.applyTheme,
        activeIndex,
        dismissOnClick,
        getItemProps,
        handleSelect
      },
      children: [
        /* @__PURE__ */ jsxRuntime.jsxs(
          Trigger,
          {
            ...buttonProps,
            refs,
            inline,
            theme: theme$1,
            "data-testid": dataTestId,
            className: tailwindMerge.twMerge(theme$1.floating.target, className),
            setButtonWidth,
            getReferenceProps,
            renderTrigger,
            children: [
              label,
              arrowIcon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.arrowIcon })
            ]
          }
        ),
        open && /* @__PURE__ */ jsxRuntime.jsx(react.FloatingFocusManager, { context, modal: false, children: /* @__PURE__ */ jsxRuntime.jsx(
          "div",
          {
            ref: refs.setFloating,
            style: { ...floatingStyles, minWidth: buttonWidth },
            "data-testid": "flowbite-dropdown",
            "aria-expanded": open,
            ...getFloatingProps({
              className: tailwindMerge.twMerge(
                theme$1.floating.base,
                theme$1.floating.animation,
                "duration-100",
                !open && theme$1.floating.hidden,
                theme$1.floating.style.auto,
                className
              )
            }),
            children: /* @__PURE__ */ jsxRuntime.jsx(react.FloatingList, { elementsRef, labelsRef, children: /* @__PURE__ */ jsxRuntime.jsx("ul", { className: theme$1.content, tabIndex: -1, children }) })
          }
        ) })
      ]
    }
  );
}
Dropdown.displayName = "Dropdown";

exports.Dropdown = Dropdown;
//# sourceMappingURL=Dropdown.cjs.map
