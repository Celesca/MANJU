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

const Timeline = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.timelineTheme, provider$1.theme?.timeline, props.theme],
    [get.get(provider$1.clearTheme, "timeline"), props.clearTheme],
    [get.get(provider$1.applyTheme, "timeline"), props.applyTheme]
  );
  const { className, horizontal, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.timeline);
  return /* @__PURE__ */ jsxRuntime.jsx(
    TimelineContext.TimelineContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, horizontal },
      children: /* @__PURE__ */ jsxRuntime.jsx(
        "ol",
        {
          ref,
          "data-testid": "timeline-component",
          className: tailwindMerge.twMerge(
            horizontal && theme$1.root.direction.horizontal,
            !horizontal && theme$1.root.direction.vertical,
            className
          ),
          ...restProps
        }
      )
    }
  );
});
Timeline.displayName = "Timeline";

exports.Timeline = Timeline;
//# sourceMappingURL=Timeline.cjs.map
