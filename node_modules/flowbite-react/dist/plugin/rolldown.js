import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';
import { pluginName } from './index.js';

var rolldown = () => ({
  name: pluginName,
  async buildStart() {
    await build();
    await dev();
  }
});

export { rolldown as default };
//# sourceMappingURL=rolldown.js.map
