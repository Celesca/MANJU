import { PHASE_DEVELOPMENT_SERVER, PHASE_PRODUCTION_BUILD } from 'next/constants.js';
import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';

let devRan = false;
let buildRan = false;
function withFlowbiteReact(nextConfig = {}) {
  return async (phase, options) => {
    if (phase === PHASE_DEVELOPMENT_SERVER && !devRan) {
      devRan = true;
      dev();
    }
    if (phase === PHASE_PRODUCTION_BUILD && !buildRan) {
      buildRan = true;
      build();
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

export { withFlowbiteReact as default };
//# sourceMappingURL=nextjs.js.map
