'use strict';

var jsxRuntime = require('react/jsx-runtime');
var index = require('../store/init/index.cjs');

function ThemeConfig({ mode }) {
  return /* @__PURE__ */ jsxRuntime.jsx(index.StoreInit, { ...{ mode } });
}
ThemeConfig.displayName = "ThemeConfig";

exports.ThemeConfig = ThemeConfig;
//# sourceMappingURL=config.cjs.map
