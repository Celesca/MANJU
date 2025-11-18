import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginRollup(configPath) {
  await addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "rollup"),
    pluginName
  });
}

export { setupPluginRollup };
//# sourceMappingURL=setup-plugin-rollup.js.map
