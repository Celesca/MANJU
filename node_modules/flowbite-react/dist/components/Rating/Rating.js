'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { RatingContext } from './RatingContext.js';
import { ratingTheme } from './theme.js';

const Rating = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [ratingTheme, provider.theme?.rating, props.theme],
    [get(provider.clearTheme, "rating"), props.clearTheme],
    [get(provider.applyTheme, "rating"), props.applyTheme]
  );
  const { className, size = "sm", ...restProps } = resolveProps(props, provider.props?.rating);
  return /* @__PURE__ */ jsx(
    RatingContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, size },
      children: /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.root.base, className), ...restProps })
    }
  );
});
Rating.displayName = "Rating";

export { Rating };
//# sourceMappingURL=Rating.js.map
