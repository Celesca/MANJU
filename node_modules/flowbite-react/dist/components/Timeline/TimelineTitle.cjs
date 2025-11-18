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

const TimelineTitle = React.forwardRef((props, ref) => {
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
      theme.timelineTheme.item.content.title,
      provider$1.theme?.timeline?.item?.content?.title,
      rootTheme?.item?.content?.title,
      itemTheme?.content?.title,
      contentTheme?.title,
      props.theme
    ],
    [
      get.get(provider$1.clearTheme, "timeline.item.content.title"),
      get.get(rootClearTheme, "item.content.title"),
      get.get(itemClearTheme, "content.title"),
      get.get(contentClearTheme, "title"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "timeline.item.content.title"),
      get.get(rootApplyTheme, "item.content.title"),
      get.get(itemApplyTheme, "content.title"),
      get.get(contentApplyTheme, "title"),
      props.applyTheme
    ]
  );
  const { as: Component = "h3", className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timelineTitle);
  return /* @__PURE__ */ jsxRuntime.jsx(Component, { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
TimelineTitle.displayName = "TimelineTitle";

exports.TimelineTitle = TimelineTitle;
//# sourceMappingURL=TimelineTitle.cjs.map
