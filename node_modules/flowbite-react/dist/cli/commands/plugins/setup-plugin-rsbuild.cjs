'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginRsbuild(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "rsbuild"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginRsbuild = setupPluginRsbuild;
//# sourceMappingURL=setup-plugin-rsbuild.cjs.map
