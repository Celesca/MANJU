import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginRolldown(configPath) {
  await addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "rolldown"),
    pluginName
  });
}

export { setupPluginRolldown };
//# sourceMappingURL=setup-plugin-rolldown.js.map
