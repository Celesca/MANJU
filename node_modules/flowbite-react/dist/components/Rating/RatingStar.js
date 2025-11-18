'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { StarIcon } from '../../icons/star-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useRatingContext } from './RatingContext.js';
import { ratingTheme } from './theme.js';

const RatingStar = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, size = "sm" } = useRatingContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [ratingTheme.star, provider.theme?.rating?.star, rootTheme?.star, props.theme],
    [get(provider.clearTheme, "rating.star"), get(rootClearTheme, "star"), props.clearTheme],
    [get(provider.applyTheme, "rating.star"), get(rootApplyTheme, "star"), props.applyTheme]
  );
  const {
    className,
    filled = true,
    starIcon: Icon = StarIcon,
    ...restProps
  } = resolveProps(props, provider.props?.ratingStar);
  return /* @__PURE__ */ jsx(
    Icon,
    {
      ref,
      "data-testid": "flowbite-rating-star",
      className: twMerge(theme.sizes[size], theme[filled ? "filled" : "empty"], className),
      ...restProps
    }
  );
});
RatingStar.displayName = "RatingStar";

export { RatingStar };
//# sourceMappingURL=RatingStar.js.map
