import fs__default from 'fs/promises';
import toml from '@iarna/toml';
import { pluginPath, pluginName } from '../../consts.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';
import { updateBuildConfig } from '../../utils/update-build-config.js';

async function setupPluginBun(configPath) {
  var _a, _b;
  try {
    const bunfig = await fs__default.readFile(configPath, "utf-8");
    const bunfigContent = toml.parse(bunfig);
    const bunPluginPath = joinNormalizedPath(pluginPath, "bun");
    if (bunfigContent.serve?.static?.plugins?.includes(pluginName)) {
      return;
    }
    bunfigContent.serve ?? (bunfigContent.serve = {});
    (_a = bunfigContent.serve).static ?? (_a.static = {});
    (_b = bunfigContent.serve.static).plugins ?? (_b.plugins = []);
    if (!bunfigContent.serve.static.plugins.includes(bunPluginPath)) {
      bunfigContent.serve.static.plugins.push(bunPluginPath);
      console.log(`Updating ${configPath} with flowbite-react configuration...`);
      await fs__default.writeFile(configPath, toml.stringify(bunfigContent), "utf-8");
    }
    const buildConfigPath = "build.ts";
    const buildConfig = await fs__default.readFile(buildConfigPath, "utf-8");
    const updatedBuildConfig = updateBuildConfig({
      content: buildConfig,
      pluginName,
      pluginImportPath: bunPluginPath
    });
    if (buildConfig !== updatedBuildConfig) {
      console.log(`Updating ${buildConfigPath} with flowbite-react configuration...`);
      await fs__default.writeFile(buildConfigPath, updatedBuildConfig, "utf-8");
    }
  } catch (error) {
    console.error(`Failed to update ${configPath} file...`, error);
  }
}

export { setupPluginBun };
//# sourceMappingURL=setup-plugin-bun.js.map
