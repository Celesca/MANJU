'use strict';

var fs = require('fs/promises');
var consts = require('../../consts.cjs');
var addImport = require('../../utils/add-import.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');
var wrapDefaultExport = require('../../utils/wrap-default-export.cjs');

async function setupPluginNextjs(configPath) {
  const pluginName = "withFlowbiteReact";
  try {
    const content = await fs.readFile(configPath, "utf-8");
    let updatedContent = addImport.addImport({
      content,
      importName: pluginName,
      importPath: normalizePath.joinNormalizedPath(consts.pluginPath, "nextjs")
    });
    if (!content.includes(`${pluginName}(`)) {
      updatedContent = wrapDefaultExport.wrapDefaultExport(updatedContent, pluginName);
    }
    if (updatedContent !== content) {
      console.log(`Updating ${configPath} with ${pluginName} plugin...`);
      await fs.writeFile(configPath, updatedContent, "utf-8");
    }
  } catch (error) {
    console.error(`Failed to setup ${pluginName} plugin:`, error);
  }
}

exports.setupPluginNextjs = setupPluginNextjs;
//# sourceMappingURL=setup-plugin-nextjs.cjs.map
