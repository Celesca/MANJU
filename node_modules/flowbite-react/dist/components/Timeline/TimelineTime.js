'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { timelineTheme } from './theme.js';
import { useTimelineContentContext } from './TimelineContentContext.js';
import { useTimelineContext } from './TimelineContext.js';
import { useTimelineItemContext } from './TimelineItemContext.js';

const TimelineTime = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useTimelineContext();
  const { theme: itemTheme, clearTheme: itemClearTheme, applyTheme: itemApplyTheme } = useTimelineItemContext();
  const {
    theme: contentTheme,
    clearTheme: contentClearTheme,
    applyTheme: contentApplyTheme
  } = useTimelineContentContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [
      timelineTheme.item.content.time,
      provider.theme?.timeline?.item?.content?.time,
      rootTheme?.item?.content?.time,
      itemTheme?.content?.time,
      contentTheme?.time,
      props.theme
    ],
    [
      get(provider.clearTheme, "timeline.item.content.time"),
      get(rootClearTheme, "item.content.time"),
      get(itemClearTheme, "content.time"),
      get(contentClearTheme, "time"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "timeline.item.content.time"),
      get(rootApplyTheme, "item.content.time"),
      get(itemApplyTheme, "content.time"),
      get(contentApplyTheme, "time"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.timelineTime);
  return /* @__PURE__ */ jsx("time", { ref, className: twMerge(theme.base, className), ...restProps });
});
TimelineTime.displayName = "TimelineTime";

export { TimelineTime };
//# sourceMappingURL=TimelineTime.js.map
