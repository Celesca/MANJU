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
var TimelineContentContext = require('./TimelineContentContext.cjs');
var TimelineContext = require('./TimelineContext.cjs');
var TimelineItemContext = require('./TimelineItemContext.cjs');

const TimelineTime = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = TimelineContext.useTimelineContext();
  const { theme: itemTheme, clearTheme: itemClearTheme, applyTheme: itemApplyTheme } = TimelineItemContext.useTimelineItemContext();
  const {
    theme: contentTheme,
    clearTheme: contentClearTheme,
    applyTheme: contentApplyTheme
  } = TimelineContentContext.useTimelineContentContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [
      theme.timelineTheme.item.content.time,
      provider$1.theme?.timeline?.item?.content?.time,
      rootTheme?.item?.content?.time,
      itemTheme?.content?.time,
      contentTheme?.time,
      props.theme
    ],
    [
      get.get(provider$1.clearTheme, "timeline.item.content.time"),
      get.get(rootClearTheme, "item.content.time"),
      get.get(itemClearTheme, "content.time"),
      get.get(contentClearTheme, "time"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "timeline.item.content.time"),
      get.get(rootApplyTheme, "item.content.time"),
      get.get(itemApplyTheme, "content.time"),
      get.get(contentApplyTheme, "time"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timelineTime);
  return /* @__PURE__ */ jsxRuntime.jsx("time", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
TimelineTime.displayName = "TimelineTime";

exports.TimelineTime = TimelineTime;
//# sourceMappingURL=TimelineTime.cjs.map
