'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginWebpack(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "webpack"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginWebpack = setupPluginWebpack;
//# sourceMappingURL=setup-plugin-webpack.cjs.map
