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

const TimelineBody = forwardRef((props, ref) => {
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
      timelineTheme.item.content.body,
      provider.theme?.timeline?.item?.content?.body,
      rootTheme?.item?.content?.body,
      itemTheme?.content?.body,
      contentTheme?.body,
      props.theme
    ],
    [
      get(provider.clearTheme, "timeline.item.content.body"),
      get(rootClearTheme, "item.content.body"),
      get(itemClearTheme, "content.body"),
      get(contentClearTheme, "body"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "timeline.item.content.body"),
      get(rootApplyTheme, "item.content.body"),
      get(itemApplyTheme, "content.body"),
      get(contentApplyTheme, "body"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.timelineBody);
  return /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.base, className), ...restProps });
});
TimelineBody.displayName = "TimelineBody";

export { TimelineBody };
//# sourceMappingURL=TimelineBody.js.map
