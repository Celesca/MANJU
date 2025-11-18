'use strict';

var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');
var index = require('./index.cjs');

var rsbuild = () => ({
  name: index.pluginName,
  setup(api) {
    api.onBeforeBuild(async () => {
      await build.build();
    });
    api.onBeforeStartDevServer(async () => {
      await dev.dev();
    });
  }
});

module.exports = rsbuild;
//# sourceMappingURL=rsbuild.cjs.map
