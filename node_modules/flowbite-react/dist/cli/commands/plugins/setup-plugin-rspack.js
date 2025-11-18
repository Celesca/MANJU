import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginRspack(configPath) {
  await addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "rspack"),
    pluginName
  });
}

export { setupPluginRspack };
//# sourceMappingURL=setup-plugin-rspack.js.map
