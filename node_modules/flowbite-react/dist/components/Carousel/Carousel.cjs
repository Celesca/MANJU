'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var dragScroll = require('../../helpers/drag-scroll.cjs');
var get = require('../../helpers/get.cjs');
var isClient = require('../../helpers/is-client.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var chevronLeftIcon = require('../../icons/chevron-left-icon.cjs');
var chevronRightIcon = require('../../icons/chevron-right-icon.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const Carousel = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.carouselTheme, provider$1.theme?.carousel, props.theme],
    [get.get(provider$1.clearTheme, "carousel"), props.clearTheme],
    [get.get(provider$1.applyTheme, "carousel"), props.applyTheme]
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
  } = resolveProps.resolveProps(props, provider$1.props?.carousel);
  const isDeviceMobile = isClient.isClient() && navigator.userAgent.indexOf("IEMobile") !== -1;
  const carouselContainer = React.useRef(null);
  const [activeItem, setActiveItem] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const [isHovering, setIsHovering] = React.useState(false);
  const didMountRef = React.useRef(false);
  const items = React.useMemo(
    () => React.Children.map(
      children,
      (child) => React.cloneElement(child, {
        className: tailwindMerge.twMerge(theme$1.item.base, child.props.className)
      })
    ),
    [children, theme$1.item.base]
  );
  const navigateTo = React.useCallback(
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
  React.useEffect(() => {
    if (carouselContainer.current && !isDragging && carouselContainer.current.scrollLeft !== 0) {
      setActiveItem(Math.round(carouselContainer.current.scrollLeft / carouselContainer.current.clientWidth));
    }
  }, [isDragging]);
  React.useEffect(() => {
    if (slide && !(pauseOnHover && isHovering)) {
      const intervalId = setInterval(() => !isDragging && navigateTo(activeItem + 1)(), slideInterval ?? 3e3);
      return () => clearInterval(intervalId);
    }
  }, [activeItem, isDragging, navigateTo, slide, slideInterval, pauseOnHover, isHovering]);
  React.useEffect(() => {
    if (didMountRef.current) {
      onSlideChange?.(activeItem);
    } else {
      didMountRef.current = true;
    }
  }, [onSlideChange, activeItem]);
  const handleDragging = (dragging) => () => setIsDragging(dragging);
  const setHoveringTrue = React.useCallback(() => setIsHovering(true), []);
  const setHoveringFalse = React.useCallback(() => setIsHovering(false), []);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "div",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.root.base, className),
      "data-testid": "carousel",
      onMouseEnter: setHoveringTrue,
      onMouseLeave: setHoveringFalse,
      onTouchStart: setHoveringTrue,
      onTouchEnd: setHoveringFalse,
      ...restProps,
      children: [
        /* @__PURE__ */ jsxRuntime.jsx(
          dragScroll,
          {
            className: tailwindMerge.twMerge(theme$1.scrollContainer.base, (isDeviceMobile || !isDragging) && theme$1.scrollContainer.snap),
            draggingClassName: "cursor-grab",
            innerRef: carouselContainer,
            onEndScroll: handleDragging(false),
            onStartScroll: handleDragging(draggable),
            vertical: false,
            horizontal: draggable,
            children: items?.map((item, index) => /* @__PURE__ */ jsxRuntime.jsx(
              "div",
              {
                className: theme$1.item.wrapper[draggable ? "on" : "off"],
                "data-active": activeItem === index,
                "data-testid": "carousel-item",
                children: item
              },
              index
            ))
          }
        ),
        indicators && /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.indicators.wrapper, children: items?.map((_, index) => /* @__PURE__ */ jsxRuntime.jsx(
          "button",
          {
            className: tailwindMerge.twMerge(theme$1.indicators.base, theme$1.indicators.active[index === activeItem ? "on" : "off"]),
            onClick: navigateTo(index),
            "data-testid": "carousel-indicator",
            "aria-label": `Slide ${index + 1}`
          },
          index
        )) }),
        items && /* @__PURE__ */ jsxRuntime.jsxs(jsxRuntime.Fragment, { children: [
          /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.root.leftControl, children: /* @__PURE__ */ jsxRuntime.jsx(
            "button",
            {
              className: "group",
              "data-testid": "carousel-left-control",
              onClick: navigateTo(activeItem - 1),
              type: "button",
              "aria-label": "Previous slide",
              children: leftControl ? leftControl : /* @__PURE__ */ jsxRuntime.jsx(DefaultLeftControl, { theme: theme$1.control })
            }
          ) }),
          /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.root.rightControl, children: /* @__PURE__ */ jsxRuntime.jsx(
            "button",
            {
              className: "group",
              "data-testid": "carousel-right-control",
              onClick: navigateTo(activeItem + 1),
              type: "button",
              "aria-label": "Next slide",
              children: rightControl ? rightControl : /* @__PURE__ */ jsxRuntime.jsx(DefaultRightControl, { theme: theme$1.control })
            }
          ) })
        ] })
      ]
    }
  );
});
Carousel.displayName = "Carousel";
function DefaultLeftControl({ theme }) {
  return /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme.base, children: /* @__PURE__ */ jsxRuntime.jsx(chevronLeftIcon.ChevronLeftIcon, { className: theme.icon }) });
}
function DefaultRightControl({ theme }) {
  return /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme.base, children: /* @__PURE__ */ jsxRuntime.jsx(chevronRightIcon.ChevronRightIcon, { className: theme.icon }) });
}

exports.Carousel = Carousel;
//# sourceMappingURL=Carousel.cjs.map
