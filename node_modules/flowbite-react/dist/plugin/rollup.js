import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';
import { pluginName } from './index.js';

var rollup = () => ({
  name: pluginName,
  async buildStart() {
    await build();
    await dev();
  }
});

export { rollup as default };
//# sourceMappingURL=rollup.js.map
