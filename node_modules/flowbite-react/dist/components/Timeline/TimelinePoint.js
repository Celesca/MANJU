'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { timelineTheme } from './theme.js';
import { useTimelineContext } from './TimelineContext.js';
import { useTimelineItemContext } from './TimelineItemContext.js';

const TimelinePoint = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, horizontal } = useTimelineContext();
  const { theme: itemTheme, clearTheme: itemClearTheme, applyTheme: itemApplyTheme } = useTimelineItemContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [
      timelineTheme.item.point,
      provider.theme?.timeline?.item?.point,
      rootTheme?.item?.point,
      itemTheme?.point,
      props.theme
    ],
    [
      get(provider.clearTheme, "timeline.item.point"),
      get(rootClearTheme, "item.point"),
      get(itemClearTheme, "point"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "timeline.item.point"),
      get(rootApplyTheme, "item.point"),
      get(itemApplyTheme, "point"),
      props.applyTheme
    ]
  );
  const { children, className, icon: Icon, ...restProps } = resolveProps(props, provider.props?.timelinePoint);
  return /* @__PURE__ */ jsxs(
    "div",
    {
      ref,
      "data-testid": "timeline-point",
      className: twMerge(horizontal && theme.horizontal, !horizontal && theme.vertical, className),
      ...restProps,
      children: [
        children,
        Icon ? /* @__PURE__ */ jsx("span", { className: twMerge(theme.marker.icon.wrapper), children: /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, className: twMerge(theme.marker.icon.base) }) }) : /* @__PURE__ */ jsx(
          "div",
          {
            className: twMerge(horizontal && theme.marker.base.horizontal, !horizontal && theme.marker.base.vertical)
          }
        ),
        horizontal && /* @__PURE__ */ jsx("div", { className: twMerge(theme.line) })
      ]
    }
  );
});
TimelinePoint.displayName = "TimelinePoint";

export { TimelinePoint };
//# sourceMappingURL=TimelinePoint.js.map
