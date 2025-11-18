'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var Tooltip = require('../Tooltip/Tooltip.cjs');
var helpers = require('./helpers.cjs');
var theme = require('./theme.cjs');

const Clipboard = React.forwardRef((props, ref) => {
  const [isJustCopied, setIsJustCopied] = React.useState(false);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.clipboardTheme.button, provider$1.theme?.clipboard?.button, props.theme],
    [get.get(provider$1.clearTheme, "clipboard.button"), props.clearTheme],
    [get.get(provider$1.applyTheme, "clipboard.button"), props.applyTheme]
  );
  const { className, valueToCopy, label, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.clipboard);
  return /* @__PURE__ */ jsxRuntime.jsx(Tooltip.Tooltip, { content: isJustCopied ? "Copied" : "Copy to clipboard", className: "[&_*]:cursor-pointer", children: /* @__PURE__ */ jsxRuntime.jsx(
    "button",
    {
      className: tailwindMerge.twMerge(theme$1.base, className),
      onClick: () => helpers.copyToClipboard(valueToCopy, setIsJustCopied),
      ...restProps,
      ref,
      children: /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.label, children: label })
    }
  ) });
});
Clipboard.displayName = "Clipboard";

exports.Clipboard = Clipboard;
//# sourceMappingURL=Clipboard.cjs.map
