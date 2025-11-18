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

const TimelineContent = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, horizontal } = TimelineContext.useTimelineContext();
  const { theme: itemTheme, clearTheme: itemClearTheme, applyTheme: itemApplyTheme } = TimelineItemContext.useTimelineItemContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [
      theme.timelineTheme.item.content,
      provider$1.theme?.timeline?.item?.content,
      rootTheme?.item?.content,
      itemTheme?.content,
      props.theme
    ],
    [
      get.get(provider$1.clearTheme, "timeline.item.content"),
      get.get(rootClearTheme, "item.content"),
      get.get(itemClearTheme, "content"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "timeline.item.content"),
      get.get(rootApplyTheme, "item.content"),
      get.get(itemApplyTheme, "content"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timelineContent);
  return /* @__PURE__ */ jsxRuntime.jsx(
    TimelineContentContext.TimelineContentContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsxRuntime.jsx(
        "div",
        {
          ref,
          "data-testid": "timeline-content",
          className: tailwindMerge.twMerge(theme$1.root.base, horizontal ? theme$1.root.horizontal : theme$1.root.vertical, className),
          ...restProps
        }
      )
    }
  );
});
TimelineContent.displayName = "TimelineContent";

exports.TimelineContent = TimelineContent;
//# sourceMappingURL=TimelineContent.cjs.map
