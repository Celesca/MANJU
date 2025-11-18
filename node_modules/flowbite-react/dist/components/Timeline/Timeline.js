'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { timelineTheme } from './theme.js';
import { TimelineContext } from './TimelineContext.js';

const Timeline = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [timelineTheme, provider.theme?.timeline, props.theme],
    [get(provider.clearTheme, "timeline"), props.clearTheme],
    [get(provider.applyTheme, "timeline"), props.applyTheme]
  );
  const { className, horizontal, ...restProps } = resolveProps(props, provider.props?.timeline);
  return /* @__PURE__ */ jsx(
    TimelineContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, horizontal },
      children: /* @__PURE__ */ jsx(
        "ol",
        {
          ref,
          "data-testid": "timeline-component",
          className: twMerge(
            horizontal && theme.root.direction.horizontal,
            !horizontal && theme.root.direction.vertical,
            className
          ),
          ...restProps
        }
      )
    }
  );
});
Timeline.displayName = "Timeline";

export { Timeline };
//# sourceMappingURL=Timeline.js.map
