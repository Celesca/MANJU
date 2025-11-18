'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var resolveProps = require('../../helpers/resolve-props.cjs');
var provider = require('../../theme/provider.cjs');
var Button = require('../Button/Button.cjs');

const BannerCollapseButton = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const mergedProps = resolveProps.resolveProps(props, provider$1.props?.bannerCollapseButton);
  function onClick(e) {
    const collapseButton = e.target;
    const parentBanner = collapseButton.closest('[role="banner"]');
    parentBanner?.remove();
  }
  return /* @__PURE__ */ jsxRuntime.jsx(Button.Button, { ref, onClick, ...mergedProps });
});
BannerCollapseButton.displayName = "BannerCollapseButton";

exports.BannerCollapseButton = BannerCollapseButton;
//# sourceMappingURL=BannerCollapseButton.cjs.map
