'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginRolldown(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "rolldown"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginRolldown = setupPluginRolldown;
//# sourceMappingURL=setup-plugin-rolldown.cjs.map
