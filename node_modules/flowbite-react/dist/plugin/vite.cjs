'use strict';

var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');
var index = require('./index.cjs');

var vitePlugin = () => ({
  name: index.pluginName,
  async buildStart() {
    await build.build();
  },
  async configureServer() {
    await dev.dev();
  }
});

module.exports = vitePlugin;
//# sourceMappingURL=vite.cjs.map
