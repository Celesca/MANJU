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

const TimelineBody = React.forwardRef((props, ref) => {
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
      theme.timelineTheme.item.content.body,
      provider$1.theme?.timeline?.item?.content?.body,
      rootTheme?.item?.content?.body,
      itemTheme?.content?.body,
      contentTheme?.body,
      props.theme
    ],
    [
      get.get(provider$1.clearTheme, "timeline.item.content.body"),
      get.get(rootClearTheme, "item.content.body"),
      get.get(itemClearTheme, "content.body"),
      get.get(contentClearTheme, "body"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "timeline.item.content.body"),
      get.get(rootApplyTheme, "item.content.body"),
      get.get(itemApplyTheme, "content.body"),
      get.get(contentApplyTheme, "body"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timelineBody);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
TimelineBody.displayName = "TimelineBody";

exports.TimelineBody = TimelineBody;
//# sourceMappingURL=TimelineBody.cjs.map
