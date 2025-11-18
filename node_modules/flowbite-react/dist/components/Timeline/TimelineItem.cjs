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
var TimelineContext = require('./TimelineContext.cjs');
var TimelineItemContext = require('./TimelineItemContext.cjs');

const TimelineItem = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, horizontal } = TimelineContext.useTimelineContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.timelineTheme.item, provider$1.theme?.timeline?.item, rootTheme?.item, props.theme],
    [get.get(provider$1.clearTheme, "timeline.item"), get.get(rootClearTheme, "item"), props.clearTheme],
    [get.get(provider$1.applyTheme, "timeline.item"), get.get(rootApplyTheme, "item"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timelineItem);
  return /* @__PURE__ */ jsxRuntime.jsx(
    TimelineItemContext.TimelineItemContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsxRuntime.jsx(
        "li",
        {
          ref,
          "data-testid": "timeline-item",
          className: tailwindMerge.twMerge(horizontal && theme$1.root.horizontal, !horizontal && theme$1.root.vertical, className),
          ...restProps
        }
      )
    }
  );
});
TimelineItem.displayName = "TimelineItem";

exports.TimelineItem = TimelineItem;
//# sourceMappingURL=TimelineItem.cjs.map
