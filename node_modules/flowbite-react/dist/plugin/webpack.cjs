'use strict';

var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');
var index = require('./index.cjs');

var webpack = () => ({
  name: index.pluginName,
  apply(compiler) {
    let devServerStarted = false;
    compiler.hooks.beforeCompile.tapPromise(index.pluginName, async () => {
      const isDev = compiler.options.mode === "development";
      const isBuild = compiler.options.mode === "production";
      if (isBuild) {
        await build.build();
      } else if (isDev && !devServerStarted) {
        devServerStarted = true;
        await build.build();
        await dev.dev();
      }
    });
  }
});

module.exports = webpack;
//# sourceMappingURL=webpack.cjs.map
