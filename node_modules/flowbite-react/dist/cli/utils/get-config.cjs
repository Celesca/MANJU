'use strict';

var setupConfig = require('../commands/setup-config.cjs');

async function getConfig() {
  return await setupConfig.setupConfig();
}

exports.getConfig = getConfig;
//# sourceMappingURL=get-config.cjs.map
