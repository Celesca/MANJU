'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef, useId } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { progressTheme } from './theme.js';

const Progress = forwardRef((props, ref) => {
  const id = useId();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [progressTheme, provider.theme?.progress, props.theme],
    [get(provider.clearTheme, "progress"), props.clearTheme],
    [get(provider.applyTheme, "progress"), props.applyTheme]
  );
  const {
    className,
    color = "default",
    labelProgress = false,
    labelText = false,
    progress,
    progressLabelPosition = "inside",
    size = "md",
    textLabel = "progressbar",
    textLabelPosition = "inside",
    ...restProps
  } = resolveProps(props, provider.props?.progress);
  return /* @__PURE__ */ jsxs("div", { ref, id, "aria-label": textLabel, "aria-valuenow": progress, role: "progressbar", ...restProps, children: [
    (textLabel && labelText && textLabelPosition === "outside" || progress > 0 && labelProgress && progressLabelPosition === "outside") && /* @__PURE__ */ jsxs("div", { className: theme.label, "data-testid": "flowbite-progress-outer-label-container", children: [
      textLabel && labelText && textLabelPosition === "outside" && /* @__PURE__ */ jsx("span", { "data-testid": "flowbite-progress-outer-text-label", children: textLabel }),
      labelProgress && progressLabelPosition === "outside" && /* @__PURE__ */ jsxs("span", { "data-testid": "flowbite-progress-outer-progress-label", children: [
        progress,
        "%"
      ] })
    ] }),
    /* @__PURE__ */ jsx("div", { className: twMerge(theme.base, theme.size[size], className), children: /* @__PURE__ */ jsxs("div", { style: { width: `${progress}%` }, className: twMerge(theme.bar, theme.color[color], theme.size[size]), children: [
      textLabel && labelText && textLabelPosition === "inside" && /* @__PURE__ */ jsx("span", { "data-testid": "flowbite-progress-inner-text-label", children: textLabel }),
      progress > 0 && labelProgress && progressLabelPosition === "inside" && /* @__PURE__ */ jsxs("span", { "data-testid": "flowbite-progress-inner-progress-label", children: [
        progress,
        "%"
      ] })
    ] }) })
  ] });
});
Progress.displayName = "Progress";

export { Progress };
//# sourceMappingURL=Progress.js.map
