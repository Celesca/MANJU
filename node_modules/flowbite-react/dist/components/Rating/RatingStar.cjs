'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var starIcon = require('../../icons/star-icon.cjs');
var provider = require('../../theme/provider.cjs');
var RatingContext = require('./RatingContext.cjs');
var theme = require('./theme.cjs');

const RatingStar = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, size = "sm" } = RatingContext.useRatingContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.ratingTheme.star, provider$1.theme?.rating?.star, rootTheme?.star, props.theme],
    [get.get(provider$1.clearTheme, "rating.star"), get.get(rootClearTheme, "star"), props.clearTheme],
    [get.get(provider$1.applyTheme, "rating.star"), get.get(rootApplyTheme, "star"), props.applyTheme]
  );
  const {
    className,
    filled = true,
    starIcon: Icon = starIcon.StarIcon,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.ratingStar);
  return /* @__PURE__ */ jsxRuntime.jsx(
    Icon,
    {
      ref,
      "data-testid": "flowbite-rating-star",
      className: tailwindMerge.twMerge(theme$1.sizes[size], theme$1[filled ? "filled" : "empty"], className),
      ...restProps
    }
  );
});
RatingStar.displayName = "RatingStar";

exports.RatingStar = RatingStar;
//# sourceMappingURL=RatingStar.cjs.map
