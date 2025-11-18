import fs__default from 'fs/promises';
import { addImport } from './add-import.js';
import { addPlugin } from './add-plugin.js';

async function addPluginToConfig({
  configKey,
  configPath,
  pluginImportPath,
  pluginName
}) {
  try {
    const content = await fs__default.readFile(configPath, "utf-8");
    const updatedContentWithImport = addImport({
      content,
      importPath: pluginImportPath,
      importName: pluginName
    });
    const finalUpdatedContent = addPlugin({
      content: updatedContentWithImport,
      targetPath: configKey,
      pluginName
    });
    if (finalUpdatedContent !== content) {
      console.log(`Updating ${configPath} with ${pluginName} plugin...`);
      await fs__default.writeFile(configPath, finalUpdatedContent, "utf-8");
    }
  } catch (error) {
    console.error(`Failed to setup ${pluginName} plugin:`, error);
  }
}

export { addPluginToConfig };
//# sourceMappingURL=add-plugin-to-config.js.map
