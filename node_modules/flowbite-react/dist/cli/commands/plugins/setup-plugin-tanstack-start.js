import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginTanStackStart(configPath) {
  await addPluginToConfig({
    configKey: "vite.plugins",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "vite"),
    pluginName
  });
}

export { setupPluginTanStackStart };
//# sourceMappingURL=setup-plugin-tanstack-start.js.map
