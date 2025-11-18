import fs__default from 'fs/promises';
import cjson from 'comment-json';
import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginModernjs(configPath) {
  await addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "modernjs"),
    pluginName
  });
  const tsConfigFile = "tsconfig.json";
  try {
    const content = await fs__default.readFile(tsConfigFile, "utf-8");
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
      await fs__default.writeFile(tsConfigFile, cjson.stringify(parsedContent, null, 2), "utf-8");
    }
  } catch (error) {
    console.error(`Failed to update ${tsConfigFile} file...`, error);
  }
}

export { setupPluginModernjs };
//# sourceMappingURL=setup-plugin-modernjs.js.map
