'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { ratingAdvancedTheme } from './theme.js';

const RatingAdvanced = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [ratingAdvancedTheme, provider.theme?.ratingAdvanced, props.theme],
    [get(provider.clearTheme, "ratingAdvanced"), props.clearTheme],
    [get(provider.applyTheme, "ratingAdvanced"), props.applyTheme]
  );
  const { children, className, percentFilled = 0, ...restProps } = resolveProps(props, provider.props?.ratingAdvanced);
  return /* @__PURE__ */ jsxs("div", { ref, className: twMerge(theme.base, className), ...restProps, children: [
    /* @__PURE__ */ jsx("span", { className: theme.label, children }),
    /* @__PURE__ */ jsx("div", { className: theme.progress.base, children: /* @__PURE__ */ jsx(
      "div",
      {
        className: theme.progress.fill,
        "data-testid": "flowbite-rating-fill",
        style: { width: `${percentFilled}%` }
      }
    ) }),
    /* @__PURE__ */ jsx("span", { className: theme.progress.label, children: `${percentFilled}%` })
  ] });
});
RatingAdvanced.displayName = "RatingAdvanced";

export { RatingAdvanced };
//# sourceMappingURL=RatingAdvanced.js.map
