'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var RatingContext = require('./RatingContext.cjs');
var theme = require('./theme.cjs');

const Rating = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.ratingTheme, provider$1.theme?.rating, props.theme],
    [get.get(provider$1.clearTheme, "rating"), props.clearTheme],
    [get.get(provider$1.applyTheme, "rating"), props.applyTheme]
  );
  const { className, size = "sm", ...restProps } = resolveProps.resolveProps(props, provider$1.props?.rating);
  return /* @__PURE__ */ jsxRuntime.jsx(
    RatingContext.RatingContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, size },
      children: /* @__PURE__ */ jsxRuntime.jsx("div", { ref, className: tailwindMerge.twMerge(theme$1.root.base, className), ...restProps })
    }
  );
});
Rating.displayName = "Rating";

exports.Rating = Rating;
//# sourceMappingURL=Rating.cjs.map
