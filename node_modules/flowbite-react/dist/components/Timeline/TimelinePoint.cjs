'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');
var TimelineContext = require('./TimelineContext.cjs');
var TimelineItemContext = require('./TimelineItemContext.cjs');

const TimelinePoint = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, horizontal } = TimelineContext.useTimelineContext();
  const { theme: itemTheme, clearTheme: itemClearTheme, applyTheme: itemApplyTheme } = TimelineItemContext.useTimelineItemContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [
      theme.timelineTheme.item.point,
      provider$1.theme?.timeline?.item?.point,
      rootTheme?.item?.point,
      itemTheme?.point,
      props.theme
    ],
    [
      get.get(provider$1.clearTheme, "timeline.item.point"),
      get.get(rootClearTheme, "item.point"),
      get.get(itemClearTheme, "point"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "timeline.item.point"),
      get.get(rootApplyTheme, "item.point"),
      get.get(itemApplyTheme, "point"),
      props.applyTheme
    ]
  );
  const { children, className, icon: Icon, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timelinePoint);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    "div",
    {
      ref,
      "data-testid": "timeline-point",
      className: tailwindMerge.twMerge(horizontal && theme$1.horizontal, !horizontal && theme$1.vertical, className),
      ...restProps,
      children: [
        children,
        Icon ? /* @__PURE__ */ jsxRuntime.jsx("span", { className: tailwindMerge.twMerge(theme$1.marker.icon.wrapper), children: /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, className: tailwindMerge.twMerge(theme$1.marker.icon.base) }) }) : /* @__PURE__ */ jsxRuntime.jsx(
          "div",
          {
            className: tailwindMerge.twMerge(horizontal && theme$1.marker.base.horizontal, !horizontal && theme$1.marker.base.vertical)
          }
        ),
        horizontal && /* @__PURE__ */ jsxRuntime.jsx("div", { className: tailwindMerge.twMerge(theme$1.line) })
      ]
    }
  );
});
TimelinePoint.displayName = "TimelinePoint";

exports.TimelinePoint = TimelinePoint;
//# sourceMappingURL=TimelinePoint.cjs.map
