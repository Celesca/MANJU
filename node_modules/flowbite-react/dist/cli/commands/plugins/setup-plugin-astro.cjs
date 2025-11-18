'use strict';

var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginAstro(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "integrations",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "astro"),
    pluginName: consts.pluginName
  });
}

exports.setupPluginAstro = setupPluginAstro;
//# sourceMappingURL=setup-plugin-astro.cjs.map
