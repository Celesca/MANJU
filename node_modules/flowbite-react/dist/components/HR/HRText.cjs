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

const HRText = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.hrTheme.text, provider$1.theme?.hr?.text, props.theme],
    [get.get(provider$1.clearTheme, "hr.text"), props.clearTheme],
    [get.get(provider$1.applyTheme, "hr.text"), props.applyTheme]
  );
  const { className, text, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.hrText);
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.base, children: [
    /* @__PURE__ */ jsxRuntime.jsx(
      "hr",
      {
        ref,
        className: tailwindMerge.twMerge(theme$1.hrLine, className),
        "data-testid": "flowbite-hr-text",
        role: "separator",
        ...restProps
      }
    ),
    /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.text, children: text })
  ] });
});
HRText.displayName = "HRText";

exports.HRText = HRText;
//# sourceMappingURL=HRText.cjs.map
