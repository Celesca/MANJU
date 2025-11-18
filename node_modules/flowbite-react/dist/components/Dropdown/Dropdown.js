'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { useListNavigation, useTypeahead, FloatingFocusManager, FloatingList } from '@floating-ui/react';
import { useState, useRef, useCallback, useMemo, useEffect, cloneElement } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useBaseFloating, useFloatingInteractions } from '../../hooks/use-floating.js';
import { ChevronDownIcon } from '../../icons/chevron-down-icon.js';
import { ChevronLeftIcon } from '../../icons/chevron-left-icon.js';
import { ChevronRightIcon } from '../../icons/chevron-right-icon.js';
import { ChevronUpIcon } from '../../icons/chevron-up-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Button } from '../Button/Button.js';
import '../Button/ButtonGroup.js';
import '../Button/ButtonGroupContext.js';
import { DropdownContext } from './DropdownContext.js';
import { dropdownTheme } from './theme.js';

const icons = {
  top: ChevronUpIcon,
  right: ChevronRightIcon,
  bottom: ChevronDownIcon,
  left: ChevronLeftIcon
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
  useEffect(() => {
    if (ref.current) {
      setButtonWidth?.(ref.current.clientWidth);
    }
  }, [ref, setButtonWidth]);
  if (renderTrigger) {
    const triggerElement = renderTrigger(theme);
    return cloneElement(triggerElement, { ref: refs.setReference, disabled, ...a11yProps, ...triggerElement.props });
  }
  return inline ? /* @__PURE__ */ jsx("button", { type: "button", ref: refs.setReference, className: theme?.inlineWrapper, disabled, ...a11yProps, children }) : /* @__PURE__ */ jsx(Button, { ...buttonProps, disabled, type: "button", ref: refs.setReference, ...a11yProps, children });
}
function Dropdown(props) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [buttonWidth, setButtonWidth] = useState(void 0);
  const elementsRef = useRef([]);
  const labelsRef = useRef([]);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [dropdownTheme, provider.theme?.dropdown, props.theme],
    [get(provider.clearTheme, "dropdown"), props.clearTheme],
    [get(provider.applyTheme, "dropdown"), props.applyTheme]
  );
  const {
    children,
    className,
    dismissOnClick = true,
    enableTypeAhead = true,
    renderTrigger,
    ...restProps
  } = resolveProps(props, provider.props?.dropdown);
  const {
    placement = restProps.inline ? "bottom-start" : "bottom",
    trigger = "click",
    label,
    inline,
    arrowIcon = true,
    ...buttonProps
  } = restProps;
  const dataTestId = restProps["data-testid"] || "flowbite-dropdown-target";
  const handleSelect = useCallback((index) => {
    setSelectedIndex(index);
    setOpen(false);
  }, []);
  const handleTypeaheadMatch = useCallback(
    (index) => {
      if (open) {
        setActiveIndex(index);
      } else {
        handleSelect(index);
      }
    },
    [open, handleSelect]
  );
  const { context, floatingStyles, refs } = useBaseFloating({
    open,
    setOpen,
    placement
  });
  const listNav = useListNavigation(context, {
    listRef: elementsRef,
    activeIndex,
    selectedIndex,
    onNavigate: setActiveIndex
  });
  const typeahead = useTypeahead(context, {
    listRef: labelsRef,
    activeIndex,
    selectedIndex,
    onMatch: handleTypeaheadMatch,
    enabled: enableTypeAhead
  });
  const { getReferenceProps, getFloatingProps, getItemProps } = useFloatingInteractions({
    context,
    role: "menu",
    trigger,
    interactions: [listNav, typeahead]
  });
  const Icon = useMemo(() => {
    const [p] = placement.split("-");
    return icons[p] ?? ChevronDownIcon;
  }, [placement]);
  return /* @__PURE__ */ jsxs(
    DropdownContext.Provider,
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
        /* @__PURE__ */ jsxs(
          Trigger,
          {
            ...buttonProps,
            refs,
            inline,
            theme,
            "data-testid": dataTestId,
            className: twMerge(theme.floating.target, className),
            setButtonWidth,
            getReferenceProps,
            renderTrigger,
            children: [
              label,
              arrowIcon && /* @__PURE__ */ jsx(Icon, { className: theme.arrowIcon })
            ]
          }
        ),
        open && /* @__PURE__ */ jsx(FloatingFocusManager, { context, modal: false, children: /* @__PURE__ */ jsx(
          "div",
          {
            ref: refs.setFloating,
            style: { ...floatingStyles, minWidth: buttonWidth },
            "data-testid": "flowbite-dropdown",
            "aria-expanded": open,
            ...getFloatingProps({
              className: twMerge(
                theme.floating.base,
                theme.floating.animation,
                "duration-100",
                !open && theme.floating.hidden,
                theme.floating.style.auto,
                className
              )
            }),
            children: /* @__PURE__ */ jsx(FloatingList, { elementsRef, labelsRef, children: /* @__PURE__ */ jsx("ul", { className: theme.content, tabIndex: -1, children }) })
          }
        ) })
      ]
    }
  );
}
Dropdown.displayName = "Dropdown";

export { Dropdown };
//# sourceMappingURL=Dropdown.js.map
