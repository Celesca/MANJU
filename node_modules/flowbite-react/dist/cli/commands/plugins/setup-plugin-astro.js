import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginAstro(configPath) {
  await addPluginToConfig({
    configKey: "integrations",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "astro"),
    pluginName
  });
}

export { setupPluginAstro };
//# sourceMappingURL=setup-plugin-astro.js.map
