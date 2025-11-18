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

const RatingAdvanced = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.ratingAdvancedTheme, provider$1.theme?.ratingAdvanced, props.theme],
    [get.get(provider$1.clearTheme, "ratingAdvanced"), props.clearTheme],
    [get.get(provider$1.applyTheme, "ratingAdvanced"), props.applyTheme]
  );
  const { children, className, percentFilled = 0, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.ratingAdvanced);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: [
    /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.label, children }),
    /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.progress.base, children: /* @__PURE__ */ jsxRuntime.jsx(
      "div",
      {
        className: theme$1.progress.fill,
        "data-testid": "flowbite-rating-fill",
        style: { width: `${percentFilled}%` }
      }
    ) }),
    /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.progress.label, children: `${percentFilled}%` })
  ] });
});
RatingAdvanced.displayName = "RatingAdvanced";

exports.RatingAdvanced = RatingAdvanced;
//# sourceMappingURL=RatingAdvanced.cjs.map
