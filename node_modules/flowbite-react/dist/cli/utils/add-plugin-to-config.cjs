'use strict';

var fs = require('fs/promises');
var addImport = require('./add-import.cjs');
var addPlugin = require('./add-plugin.cjs');

async function addPluginToConfig({
  configKey,
  configPath,
  pluginImportPath,
  pluginName
}) {
  try {
    const content = await fs.readFile(configPath, "utf-8");
    const updatedContentWithImport = addImport.addImport({
      content,
      importPath: pluginImportPath,
      importName: pluginName
    });
    const finalUpdatedContent = addPlugin.addPlugin({
      content: updatedContentWithImport,
      targetPath: configKey,
      pluginName
    });
    if (finalUpdatedContent !== content) {
      console.log(`Updating ${configPath} with ${pluginName} plugin...`);
      await fs.writeFile(configPath, finalUpdatedContent, "utf-8");
    }
  } catch (error) {
    console.error(`Failed to setup ${pluginName} plugin:`, error);
  }
}

exports.addPluginToConfig = addPluginToConfig;
//# sourceMappingURL=add-plugin-to-config.cjs.map
