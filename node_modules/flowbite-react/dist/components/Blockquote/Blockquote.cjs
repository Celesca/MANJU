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

const Blockquote = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.blockquoteTheme, provider$1.theme?.blockquote, props.theme],
    [get.get(provider$1.clearTheme, "blockquote"), props.clearTheme],
    [get.get(provider$1.applyTheme, "blockquote"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.blockquote);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "blockquote",
    {
      ref,
      className: tailwindMerge.twMerge(theme$1.root.base, className),
      "data-testid": "flowbite-blockquote",
      ...restProps
    }
  );
});
Blockquote.displayName = "Blockquote";

exports.Blockquote = Blockquote;
//# sourceMappingURL=Blockquote.cjs.map
