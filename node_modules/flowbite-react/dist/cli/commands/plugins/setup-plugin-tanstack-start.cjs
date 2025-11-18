'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginTanStackStart(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "vite.plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "vite"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginTanStackStart = setupPluginTanStackStart;
//# sourceMappingURL=setup-plugin-tanstack-start.cjs.map
