import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';
import { pluginName } from './index.js';

var farm = () => ({
  name: pluginName,
  buildStart: {
    async executor() {
      await build();
    }
  },
  async configureDevServer() {
    await dev();
  }
});

export { farm as default };
//# sourceMappingURL=farm.js.map
