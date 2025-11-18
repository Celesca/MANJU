'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginFarm(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "farm"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginFarm = setupPluginFarm;
//# sourceMappingURL=setup-plugin-farm.cjs.map
