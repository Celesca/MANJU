'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef, useState } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Tooltip } from '../Tooltip/Tooltip.js';
import { copyToClipboard } from './helpers.js';
import { clipboardTheme } from './theme.js';

const Clipboard = forwardRef((props, ref) => {
  const [isJustCopied, setIsJustCopied] = useState(false);
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [clipboardTheme.button, provider.theme?.clipboard?.button, props.theme],
    [get(provider.clearTheme, "clipboard.button"), props.clearTheme],
    [get(provider.applyTheme, "clipboard.button"), props.applyTheme]
  );
  const { className, valueToCopy, label, ...restProps } = resolveProps(props, provider.props?.clipboard);
  return /* @__PURE__ */ jsx(Tooltip, { content: isJustCopied ? "Copied" : "Copy to clipboard", className: "[&_*]:cursor-pointer", children: /* @__PURE__ */ jsx(
    "button",
    {
      className: twMerge(theme.base, className),
      onClick: () => copyToClipboard(valueToCopy, setIsJustCopied),
      ...restProps,
      ref,
      children: /* @__PURE__ */ jsx("span", { className: theme.label, children: label })
    }
  ) });
});
Clipboard.displayName = "Clipboard";

export { Clipboard };
//# sourceMappingURL=Clipboard.js.map
