'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var checkIcon = require('../../icons/check-icon.cjs');
var clipboardListIcon = require('../../icons/clipboard-list-icon.cjs');
var provider = require('../../theme/provider.cjs');
var helpers = require('./helpers.cjs');
var theme = require('./theme.cjs');

const ClipboardWithIconText = React.forwardRef((props, ref) => {
  const [isJustCopied, setIsJustCopied] = React.useState(false);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.clipboardTheme.withIconText, provider$1.theme?.clipboard?.withIconText, props.theme],
    [get.get(provider$1.clearTheme, "clipboard.withIconText"), props.clearTheme],
    [get.get(provider$1.applyTheme, "clipboard.withIconText"), props.applyTheme]
  );
  const {
    valueToCopy,
    icon: Icon = clipboardListIcon.ClipboardListIcon,
    label = "Copy",
    className,
    ...rest
  } = resolveProps.resolveProps(props, provider$1.props?.clipboardWithIconText);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "button",
    {
      className: tailwindMerge.twMerge(theme$1.base, className),
      onClick: () => helpers.copyToClipboard(valueToCopy, setIsJustCopied),
      ...rest,
      ref,
      children: isJustCopied ? /* @__PURE__ */ jsxRuntime.jsxs("span", { className: theme$1.label.base, children: [
        /* @__PURE__ */ jsxRuntime.jsx(checkIcon.CheckIcon, { "aria-hidden": true, className: theme$1.icon.successIcon }),
        /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.label.successText, children: "Copied" })
      ] }) : /* @__PURE__ */ jsxRuntime.jsxs("span", { className: theme$1.label.base, children: [
        /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, className: theme$1.icon.defaultIcon }),
        /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.label.defaultText, children: label })
      ] })
    }
  );
});
ClipboardWithIconText.displayName = "Clipboard.WithIconText";

exports.ClipboardWithIconText = ClipboardWithIconText;
//# sourceMappingURL=ClipboardWithIconText.cjs.map
