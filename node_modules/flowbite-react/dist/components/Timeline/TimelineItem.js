'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { timelineTheme } from './theme.js';
import { useTimelineContext } from './TimelineContext.js';
import { TimelineItemContext } from './TimelineItemContext.js';

const TimelineItem = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, horizontal } = useTimelineContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [timelineTheme.item, provider.theme?.timeline?.item, rootTheme?.item, props.theme],
    [get(provider.clearTheme, "timeline.item"), get(rootClearTheme, "item"), props.clearTheme],
    [get(provider.applyTheme, "timeline.item"), get(rootApplyTheme, "item"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.timelineItem);
  return /* @__PURE__ */ jsx(
    TimelineItemContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsx(
        "li",
        {
          ref,
          "data-testid": "timeline-item",
          className: twMerge(horizontal && theme.root.horizontal, !horizontal && theme.root.vertical, className),
          ...restProps
        }
      )
    }
  );
});
TimelineItem.displayName = "TimelineItem";

export { TimelineItem };
//# sourceMappingURL=TimelineItem.js.map
