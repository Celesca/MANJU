'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { forwardRef, useState } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { CheckIcon } from '../../icons/check-icon.js';
import { ClipboardListIcon } from '../../icons/clipboard-list-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { copyToClipboard } from './helpers.js';
import { clipboardTheme } from './theme.js';

const ClipboardWithIconText = forwardRef((props, ref) => {
  const [isJustCopied, setIsJustCopied] = useState(false);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [clipboardTheme.withIconText, provider.theme?.clipboard?.withIconText, props.theme],
    [get(provider.clearTheme, "clipboard.withIconText"), props.clearTheme],
    [get(provider.applyTheme, "clipboard.withIconText"), props.applyTheme]
  );
  const {
    valueToCopy,
    icon: Icon = ClipboardListIcon,
    label = "Copy",
    className,
    ...rest
  } = resolveProps(props, provider.props?.clipboardWithIconText);
  return /* @__PURE__ */ jsx(
    "button",
    {
      className: twMerge(theme.base, className),
      onClick: () => copyToClipboard(valueToCopy, setIsJustCopied),
      ...rest,
      ref,
      children: isJustCopied ? /* @__PURE__ */ jsxs("span", { className: theme.label.base, children: [
        /* @__PURE__ */ jsx(CheckIcon, { "aria-hidden": true, className: theme.icon.successIcon }),
        /* @__PURE__ */ jsx("span", { className: theme.label.successText, children: "Copied" })
      ] }) : /* @__PURE__ */ jsxs("span", { className: theme.label.base, children: [
        /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, className: theme.icon.defaultIcon }),
        /* @__PURE__ */ jsx("span", { className: theme.label.defaultText, children: label })
      ] })
    }
  );
});
ClipboardWithIconText.displayName = "Clipboard.WithIconText";

export { ClipboardWithIconText };
//# sourceMappingURL=ClipboardWithIconText.js.map
