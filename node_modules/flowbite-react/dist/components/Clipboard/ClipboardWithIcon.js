'use client';
import { jsx } from 'react/jsx-runtime';
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

const ClipboardWithIcon = forwardRef((props, ref) => {
  const [isJustCopied, setIsJustCopied] = useState(false);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [clipboardTheme.withIcon, provider.theme?.clipboard?.withIcon, props.theme],
    [get(provider.clearTheme, "clipboard.withIcon"), props.clearTheme],
    [get(provider.applyTheme, "clipboard.withIcon"), props.applyTheme]
  );
  const {
    valueToCopy,
    icon: Icon = ClipboardListIcon,
    className,
    ...restProps
  } = resolveProps(props, provider.props?.clipboardWithIcon);
  return /* @__PURE__ */ jsx(
    "button",
    {
      className: twMerge(theme.base, className),
      onClick: () => copyToClipboard(valueToCopy, setIsJustCopied),
      ...restProps,
      ref,
      children: isJustCopied ? /* @__PURE__ */ jsx(CheckIcon, { "aria-hidden": true, className: theme.icon.successIcon }) : /* @__PURE__ */ jsx(Icon, { "aria-hidden": true, className: theme.icon.defaultIcon })
    }
  );
});
ClipboardWithIcon.displayName = "Clipboard.WithIcon";

export { ClipboardWithIcon };
//# sourceMappingURL=ClipboardWithIcon.js.map
