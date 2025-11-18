'use strict';

var constants_js = require('next/constants.js');
var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');

let devRan = false;
let buildRan = false;
function withFlowbiteReact(nextConfig = {}) {
  return async (phase, options) => {
    if (phase === constants_js.PHASE_DEVELOPMENT_SERVER && !devRan) {
      devRan = true;
      dev.dev();
    }
    if (phase === constants_js.PHASE_PRODUCTION_BUILD && !buildRan) {
      buildRan = true;
      build.build();
    }
    const userConfig = typeof nextConfig === "function" ? await nextConfig(phase, options) : nextConfig;
    return patchConfig(userConfig);
  };
}
function patchConfig(config) {
  const experimental = config.experimental || {};
  const optimizePackageImports = experimental.optimizePackageImports || [];
  return {
    ...config,
    experimental: {
      ...experimental,
      optimizePackageImports: [...optimizePackageImports, "flowbite-react"]
    }
  };
}

module.exports = withFlowbiteReact;
//# sourceMappingURL=nextjs.cjs.map
