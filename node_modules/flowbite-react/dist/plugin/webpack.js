import { build } from '../cli/commands/build.js';
import { dev } from '../cli/commands/dev.js';
import { pluginName } from './index.js';

var webpack = () => ({
  name: pluginName,
  apply(compiler) {
    let devServerStarted = false;
    compiler.hooks.beforeCompile.tapPromise(pluginName, async () => {
      const isDev = compiler.options.mode === "development";
      const isBuild = compiler.options.mode === "production";
      if (isBuild) {
        await build();
      } else if (isDev && !devServerStarted) {
        devServerStarted = true;
        await build();
        await dev();
      }
    });
  }
});

export { webpack as default };
//# sourceMappingURL=webpack.js.map
