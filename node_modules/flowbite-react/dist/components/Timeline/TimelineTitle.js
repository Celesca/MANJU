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

const TimelineTitle = forwardRef((props, ref) => {
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
      timelineTheme.item.content.title,
      provider.theme?.timeline?.item?.content?.title,
      rootTheme?.item?.content?.title,
      itemTheme?.content?.title,
      contentTheme?.title,
      props.theme
    ],
    [
      get(provider.clearTheme, "timeline.item.content.title"),
      get(rootClearTheme, "item.content.title"),
      get(itemClearTheme, "content.title"),
      get(contentClearTheme, "title"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "timeline.item.content.title"),
      get(rootApplyTheme, "item.content.title"),
      get(itemApplyTheme, "content.title"),
      get(contentApplyTheme, "title"),
      props.applyTheme
    ]
  );
  const { as: Component = "h3", className, ...restProps } = resolveProps(props, provider.props?.timelineTitle);
  return /* @__PURE__ */ jsx(Component, { ref, className: twMerge(theme.base, className), ...restProps });
});
TimelineTitle.displayName = "TimelineTitle";

export { TimelineTitle };
//# sourceMappingURL=TimelineTitle.js.map
