'use strict';

var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');
var index = require('./index.cjs');

var farm = () => ({
  name: index.pluginName,
  buildStart: {
    async executor() {
      await build.build();
    }
  },
  async configureDevServer() {
    await dev.dev();
  }
});

module.exports = farm;
//# sourceMappingURL=farm.cjs.map
