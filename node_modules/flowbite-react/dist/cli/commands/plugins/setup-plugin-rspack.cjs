'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginRspack(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "rspack"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginRspack = setupPluginRspack;
//# sourceMappingURL=setup-plugin-rspack.cjs.map
