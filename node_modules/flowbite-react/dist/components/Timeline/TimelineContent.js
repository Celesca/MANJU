'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { timelineTheme } from './theme.js';
import { TimelineContentContext } from './TimelineContentContext.js';
import { useTimelineContext } from './TimelineContext.js';
import { useTimelineItemContext } from './TimelineItemContext.js';

const TimelineContent = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, horizontal } = useTimelineContext();
  const { theme: itemTheme, clearTheme: itemClearTheme, applyTheme: itemApplyTheme } = useTimelineItemContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [
      timelineTheme.item.content,
      provider.theme?.timeline?.item?.content,
      rootTheme?.item?.content,
      itemTheme?.content,
      props.theme
    ],
    [
      get(provider.clearTheme, "timeline.item.content"),
      get(rootClearTheme, "item.content"),
      get(itemClearTheme, "content"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "timeline.item.content"),
      get(rootApplyTheme, "item.content"),
      get(itemApplyTheme, "content"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.timelineContent);
  return /* @__PURE__ */ jsx(
    TimelineContentContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsx(
        "div",
        {
          ref,
          "data-testid": "timeline-content",
          className: twMerge(theme.root.base, horizontal ? theme.root.horizontal : theme.root.vertical, className),
          ...restProps
        }
      )
    }
  );
});
TimelineContent.displayName = "TimelineContent";

export { TimelineContent };
//# sourceMappingURL=TimelineContent.js.map
