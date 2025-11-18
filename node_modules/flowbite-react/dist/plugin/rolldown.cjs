'use strict';

var build = require('../cli/commands/build.cjs');
var dev = require('../cli/commands/dev.cjs');
var index = require('./index.cjs');

var rolldown = () => ({
  name: index.pluginName,
  async buildStart() {
    await build.build();
    await dev.dev();
  }
});

module.exports = rolldown;
//# sourceMappingURL=rolldown.cjs.map
