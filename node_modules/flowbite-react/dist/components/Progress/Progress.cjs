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

const Progress = React.forwardRef((props, ref) => {
  const id = React.useId();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.progressTheme, provider$1.theme?.progress, props.theme],
    [get.get(provider$1.clearTheme, "progress"), props.clearTheme],
    [get.get(provider$1.applyTheme, "progress"), props.applyTheme]
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
  } = resolveProps.resolveProps(props, provider$1.props?.progress);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { ref, id, "aria-label": textLabel, "aria-valuenow": progress, role: "progressbar", ...restProps, children: [
    (textLabel && labelText && textLabelPosition === "outside" || progress > 0 && labelProgress && progressLabelPosition === "outside") && /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.label, "data-testid": "flowbite-progress-outer-label-container", children: [
      textLabel && labelText && textLabelPosition === "outside" && /* @__PURE__ */ jsxRuntime.jsx("span", { "data-testid": "flowbite-progress-outer-text-label", children: textLabel }),
      labelProgress && progressLabelPosition === "outside" && /* @__PURE__ */ jsxRuntime.jsxs("span", { "data-testid": "flowbite-progress-outer-progress-label", children: [
        progress,
        "%"
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntime.jsx("div", { className: tailwindMerge.twMerge(theme$1.base, theme$1.size[size], className), children: /* @__PURE__ */ jsxRuntime.jsxs("div", { style: { width: `${progress}%` }, className: tailwindMerge.twMerge(theme$1.bar, theme$1.color[color], theme$1.size[size]), children: [
      textLabel && labelText && textLabelPosition === "inside" && /* @__PURE__ */ jsxRuntime.jsx("span", { "data-testid": "flowbite-progress-inner-text-label", children: textLabel }),
      progress > 0 && labelProgress && progressLabelPosition === "inside" && /* @__PURE__ */ jsxRuntime.jsxs("span", { "data-testid": "flowbite-progress-inner-progress-label", children: [
        progress,
        "%"
      ] })
    ] }) })
  ] });
});
Progress.displayName = "Progress";

exports.Progress = Progress;
//# sourceMappingURL=Progress.cjs.map
