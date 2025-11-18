import { pluginPath, pluginName } from '../../consts.js';
import { addPluginToConfig } from '../../utils/add-plugin-to-config.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginRsbuild(configPath) {
  await addPluginToConfig({
    configKey: "plugins",
    configPath,
    pluginImportPath: joinNormalizedPath(pluginPath, "rsbuild"),
    pluginName
  });
}

export { setupPluginRsbuild };
//# sourceMappingURL=setup-plugin-rsbuild.js.map
