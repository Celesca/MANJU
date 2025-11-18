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

const FileInput = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.fileInputTheme, provider$1.theme?.fileInput, props.theme],
    [get.get(provider$1.clearTheme, "fileInput"), props.clearTheme],
    [get.get(provider$1.applyTheme, "fileInput"), props.applyTheme]
  );
  const { className, color = "gray", sizing = "md", ...restProps } = resolveProps.resolveProps(props, provider$1.props?.fileInput);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "input",
    {
      ref,
      type: "file",
      className: tailwindMerge.twMerge(theme$1.base, theme$1.colors[color], theme$1.sizes[sizing], className),
      ...restProps
    }
  );
});
FileInput.displayName = "FileInput";

exports.FileInput = FileInput;
//# sourceMappingURL=FileInput.cjs.map
