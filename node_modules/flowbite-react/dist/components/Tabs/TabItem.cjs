'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var resolveProps = require('../../helpers/resolve-props.cjs');
require('../MegaMenu/theme.cjs');
require('../../theme/config.cjs');
require('../../theme/mode-script.cjs');
var provider = require('../../theme/provider.cjs');

const TabItem = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const { title: _, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.tabItem);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, ...restProps });
});
TabItem.displayName = "TabItem";

exports.TabItem = TabItem;
//# sourceMappingURL=TabItem.cjs.map
