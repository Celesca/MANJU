'use strict';

var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');
var index = require('./index.cjs');

var rollup = () => ({
  name: index.pluginName,
  async buildStart() {
    await build.build();
    await dev.dev();
  }
});

module.exports = rollup;
//# sourceMappingURL=rollup.cjs.map
