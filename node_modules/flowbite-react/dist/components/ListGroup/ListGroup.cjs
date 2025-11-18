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

const ListGroup = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.listGroupTheme.root, provider$1.theme?.listGroup?.root, props.theme],
    [get.get(provider$1.clearTheme, "listGroup.root"), props.clearTheme],
    [get.get(provider$1.applyTheme, "listGroup.root"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.listGroup);
  return /* @__PURE__ */ jsxRuntime.jsx("ul", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
ListGroup.displayName = "ListGroup";

exports.ListGroup = ListGroup;
//# sourceMappingURL=ListGroup.cjs.map
