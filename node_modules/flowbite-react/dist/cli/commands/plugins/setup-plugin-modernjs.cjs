'use strict';

var fs = require('fs/promises');
var cjson = require('comment-json');
var consts = require('../../consts.cjs');
var addPluginToConfig = require('../../utils/add-plugin-to-config.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginModernjs(configPath) {
  await addPluginToConfig.addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: normalizePath.joinNormalizedPath(consts.pluginPath, "modernjs"),
    pluginName: consts.pluginName
  });
  const tsConfigFile = "tsconfig.json";
  try {
    const content = await fs.readFile(tsConfigFile, "utf-8");
    const parsedContent = cjson.parse(content);
    const defaultTsConfig = {
      module: "ESNext",
      moduleResolution: "bundler"
    };
    if (parsedContent.compilerOptions?.module !== defaultTsConfig.module || parsedContent.compilerOptions?.moduleResolution !== defaultTsConfig.moduleResolution) {
      parsedContent.compilerOptions || (parsedContent.compilerOptions = {});
      parsedContent.compilerOptions.module = "ESNext";
      parsedContent.compilerOptions.moduleResolution = "bundler";
      console.log(`Updating ${tsConfigFile} file with flowbite-react configuration...`);
      await fs.writeFile(tsConfigFile, cjson.stringify(parsedContent, null, 2), "utf-8");
    }
  } catch (error) {
    console.error(`Failed to update ${tsConfigFile} file...`, error);
  }
}

exports.setupPluginModernjs = setupPluginModernjs;
//# sourceMappingURL=setup-plugin-modernjs.cjs.map
