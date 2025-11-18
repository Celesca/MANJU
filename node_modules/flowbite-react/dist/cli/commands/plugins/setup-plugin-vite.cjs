'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginVite(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "vite"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginVite = setupPluginVite;
//# sourceMappingURL=setup-plugin-vite.cjs.map
