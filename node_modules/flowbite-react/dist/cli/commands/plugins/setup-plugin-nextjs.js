import fs__default from 'fs/promises';
import { pluginPath } from '../../consts.js';
import { addImport } from '../../utils/add-import.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';
import { wrapDefaultExport } from '../../utils/wrap-default-export.js';

async function setupPluginNextjs(configPath) {
  const pluginName = "withFlowbiteReact";
  try {
    const content = await fs__default.readFile(configPath, "utf-8");
    let updatedContent = addImport({
      content,
      importName: pluginName,
      importPath: joinNormalizedPath(pluginPath, "nextjs")
    });
    if (!content.includes(`${pluginName}(`)) {
      updatedContent = wrapDefaultExport(updatedContent, pluginName);
    }
    if (updatedContent !== content) {
      console.log(`Updating ${configPath} with ${pluginName} plugin...`);
      await fs__default.writeFile(configPath, updatedContent, "utf-8");
    }
  } catch (error) {
    console.error(`Failed to setup ${pluginName} plugin:`, error);
  }
}

export { setupPluginNextjs };
//# sourceMappingURL=setup-plugin-nextjs.js.map
