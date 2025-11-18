'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginRollup(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "rollup"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginRollup = setupPluginRollup;
//# sourceMappingURL=setup-plugin-rollup.cjs.map
