'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var resolveProps = require('../../helpers/resolve-props.cjs');
var provider = require('../../theme/provider.cjs');

const Banner = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const mergedProps = resolveProps.resolveProps(props, provider$1.props?.banner);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, "data-testid": "flowbite-banner", role: "banner", tabIndex: -1, ...mergedProps });
});
Banner.displayName = "Banner";

exports.Banner = Banner;
//# sourceMappingURL=Banner.cjs.map
