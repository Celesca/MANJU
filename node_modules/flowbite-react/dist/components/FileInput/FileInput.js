'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { fileInputTheme } from './theme.js';

const FileInput = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [fileInputTheme, provider.theme?.fileInput, props.theme],
    [get(provider.clearTheme, "fileInput"), props.clearTheme],
    [get(provider.applyTheme, "fileInput"), props.applyTheme]
  );
  const { className, color = "gray", sizing = "md", ...restProps } = resolveProps(props, provider.props?.fileInput);
  return /* @__PURE__ */ jsx(
    "input",
    {
      ref,
      type: "file",
      className: twMerge(theme.base, theme.colors[color], theme.sizes[sizing], className),
      ...restProps
    }
  );
});
FileInput.displayName = "FileInput";

export { FileInput };
//# sourceMappingURL=FileInput.js.map
