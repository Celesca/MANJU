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

const ClipboardWithIcon = React.forwardRef((props, ref) => {
  const [isJustCopied, setIsJustCopied] = React.useState(false);
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.clipboardTheme.withIcon, provider$1.theme?.clipboard?.withIcon, props.theme],
    [get.get(provider$1.clearTheme, "clipboard.withIcon"), props.clearTheme],
    [get.get(provider$1.applyTheme, "clipboard.withIcon"), props.applyTheme]
  );
  const {
    valueToCopy,
    icon: Icon = clipboardListIcon.ClipboardListIcon,
    className,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.clipboardWithIcon);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "button",
    {
      className: tailwindMerge.twMerge(theme$1.base, className),
      onClick: () => helpers.copyToClipboard(valueToCopy, setIsJustCopied),
      ...restProps,
      ref,
      children: isJustCopied ? /* @__PURE__ */ jsxRuntime.jsx(checkIcon.CheckIcon, { "aria-hidden": true, className: theme$1.icon.successIcon }) : /* @__PURE__ */ jsxRuntime.jsx(Icon, { "aria-hidden": true, className: theme$1.icon.defaultIcon })
    }
  );
});
ClipboardWithIcon.displayName = "Clipboard.WithIcon";

exports.ClipboardWithIcon = ClipboardWithIcon;
//# sourceMappingURL=ClipboardWithIcon.cjs.map
