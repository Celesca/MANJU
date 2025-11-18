import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';
import { pluginName } from './index.js';

var rsbuild = () => ({
  name: pluginName,
  setup(api) {
    api.onBeforeBuild(async () => {
      await build();
    });
    api.onBeforeStartDevServer(async () => {
      await dev();
    });
  }
});

export { rsbuild as default };
//# sourceMappingURL=rsbuild.js.map
