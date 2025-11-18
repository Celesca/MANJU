import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';
import { pluginName } from './index.js';

var vitePlugin = () => ({
  name: pluginName,
  async buildStart() {
    await build();
  },
  async configureServer() {
    await dev();
  }
});

export { vitePlugin as default };
//# sourceMappingURL=vite.js.map
