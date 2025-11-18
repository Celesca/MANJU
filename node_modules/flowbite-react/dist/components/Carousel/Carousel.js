'use client';
import { jsxs, jsx, Fragment } from 'react/jsx-runtime';
import { forwardRef, useRef, useState, useMemo, Children, cloneElement, useCallback, useEffect } from 'react';
import ScrollContainer from '../../helpers/drag-scroll.js';
import { get } from '../../helpers/get.js';
import { isClient } from '../../helpers/is-client.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { ChevronLeftIcon } from '../../icons/chevron-left-icon.js';
import { ChevronRightIcon } from '../../icons/chevron-right-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { carouselTheme } from './theme.js';

const Carousel = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [carouselTheme, provider.theme?.carousel, props.theme],
    [get(provider.clearTheme, "carousel"), props.clearTheme],
    [get(provider.applyTheme, "carousel"), props.applyTheme]
  );
  const {
    children,
    indicators = true,
    leftControl,
    rightControl,
    slide = true,
    draggable = true,
    slideInterval,
    className,
    onSlideChange,
    pauseOnHover = false,
    ...restProps
  } = resolveProps(props, provider.props?.carousel);
  const isDeviceMobile = isClient() && navigator.userAgent.indexOf("IEMobile") !== -1;
  const carouselContainer = useRef(null);
  const [activeItem, setActiveItem] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const didMountRef = useRef(false);
  const items = useMemo(
    () => Children.map(
      children,
      (child) => cloneElement(child, {
        className: twMerge(theme.item.base, child.props.className)
      })
    ),
    [children, theme.item.base]
  );
  const navigateTo = useCallback(
    (item) => () => {
      if (!items) return;
      item = (item + items.length) % items.length;
      if (carouselContainer.current) {
        carouselContainer.current.scrollLeft = carouselContainer.current.clientWidth * item;
      }
      setActiveItem(item);
    },
    [items]
  );
  useEffect(() => {
    if (carouselContainer.current && !isDragging && carouselContainer.current.scrollLeft !== 0) {
      setActiveItem(Math.round(carouselContainer.current.scrollLeft / carouselContainer.current.clientWidth));
    }
  }, [isDragging]);
  useEffect(() => {
    if (slide && !(pauseOnHover && isHovering)) {
      const intervalId = setInterval(() => !isDragging && navigateTo(activeItem + 1)(), slideInterval ?? 3e3);
      return () => clearInterval(intervalId);
    }
  }, [activeItem, isDragging, navigateTo, slide, slideInterval, pauseOnHover, isHovering]);
  useEffect(() => {
    if (didMountRef.current) {
      onSlideChange?.(activeItem);
    } else {
      didMountRef.current = true;
    }
  }, [onSlideChange, activeItem]);
  const handleDragging = (dragging) => () => setIsDragging(dragging);
  const setHoveringTrue = useCallback(() => setIsHovering(true), []);
  const setHoveringFalse = useCallback(() => setIsHovering(false), []);
  return /* @__PURE__ */ jsxs(
    "div",
    {
      ref,
      className: twMerge(theme.root.base, className),
      "data-testid": "carousel",
      onMouseEnter: setHoveringTrue,
      onMouseLeave: setHoveringFalse,
      onTouchStart: setHoveringTrue,
      onTouchEnd: setHoveringFalse,
      ...restProps,
      children: [
        /* @__PURE__ */ jsx(
          ScrollContainer,
          {
            className: twMerge(theme.scrollContainer.base, (isDeviceMobile || !isDragging) && theme.scrollContainer.snap),
            draggingClassName: "cursor-grab",
            innerRef: carouselContainer,
            onEndScroll: handleDragging(false),
            onStartScroll: handleDragging(draggable),
            vertical: false,
            horizontal: draggable,
            children: items?.map((item, index) => /* @__PURE__ */ jsx(
              "div",
              {
                className: theme.item.wrapper[draggable ? "on" : "off"],
                "data-active": activeItem === index,
                "data-testid": "carousel-item",
                children: item
              },
              index
            ))
          }
        ),
        indicators && /* @__PURE__ */ jsx("div", { className: theme.indicators.wrapper, children: items?.map((_, index) => /* @__PURE__ */ jsx(
          "button",
          {
            className: twMerge(theme.indicators.base, theme.indicators.active[index === activeItem ? "on" : "off"]),
            onClick: navigateTo(index),
            "data-testid": "carousel-indicator",
            "aria-label": `Slide ${index + 1}`
          },
          index
        )) }),
        items && /* @__PURE__ */ jsxs(Fragment, { children: [
          /* @__PURE__ */ jsx("div", { className: theme.root.leftControl, children: /* @__PURE__ */ jsx(
            "button",
            {
              className: "group",
              "data-testid": "carousel-left-control",
              onClick: navigateTo(activeItem - 1),
              type: "button",
              "aria-label": "Previous slide",
              children: leftControl ? leftControl : /* @__PURE__ */ jsx(DefaultLeftControl, { theme: theme.control })
            }
          ) }),
          /* @__PURE__ */ jsx("div", { className: theme.root.rightControl, children: /* @__PURE__ */ jsx(
            "button",
            {
              className: "group",
              "data-testid": "carousel-right-control",
              onClick: navigateTo(activeItem + 1),
              type: "button",
              "aria-label": "Next slide",
              children: rightControl ? rightControl : /* @__PURE__ */ jsx(DefaultRightControl, { theme: theme.control })
            }
          ) })
        ] })
      ]
    }
  );
});
Carousel.displayName = "Carousel";
function DefaultLeftControl({ theme }) {
  return /* @__PURE__ */ jsx("span", { className: theme.base, children: /* @__PURE__ */ jsx(ChevronLeftIcon, { className: theme.icon }) });
}
function DefaultRightControl({ theme }) {
  return /* @__PURE__ */ jsx("span", { className: theme.base, children: /* @__PURE__ */ jsx(ChevronRightIcon, { className: theme.icon }) });
}

export { Carousel };
//# sourceMappingURL=Carousel.js.map
