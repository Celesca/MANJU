'use strict';

var fs = require('fs/promises');
var toml = require('@iarna/toml');
var consts = require('../../consts.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');
var updateBuildConfig = require('../../utils/update-build-config.cjs');

async function setupPluginBun(configPath) {
  var _a, _b;
  try {
    const bunfig = await fs.readFile(configPath, "utf-8");
    const bunfigContent = toml.parse(bunfig);
    const bunPluginPath = normalizePath.joinNormalizedPath(consts.pluginPath, "bun");
    if (bunfigContent.serve?.static?.plugins?.includes(consts.pluginName)) {
      return;
    }
    bunfigContent.serve ?? (bunfigContent.serve = {});
    (_a = bunfigContent.serve).static ?? (_a.static = {});
    (_b = bunfigContent.serve.static).plugins ?? (_b.plugins = []);
    if (!bunfigContent.serve.static.plugins.includes(bunPluginPath)) {
      bunfigContent.serve.static.plugins.push(bunPluginPath);
      console.log(`Updating ${configPath} with flowbite-react configuration...`);
      await fs.writeFile(configPath, toml.stringify(bunfigContent), "utf-8");
    }
    const buildConfigPath = "build.ts";
    const buildConfig = await fs.readFile(buildConfigPath, "utf-8");
    const updatedBuildConfig = updateBuildConfig.updateBuildConfig({
      content: buildConfig,
      pluginName: consts.pluginName,
      pluginImportPath: bunPluginPath
    });
    if (buildConfig !== updatedBuildConfig) {
      console.log(`Updating ${buildConfigPath} with flowbite-react configuration...`);
      await fs.writeFile(buildConfigPath, updatedBuildConfig, "utf-8");
    }
  } catch (error) {
    console.error(`Failed to update ${configPath} file...`, error);
  }
}

exports.setupPluginBun = setupPluginBun;
//# sourceMappingURL=setup-plugin-bun.cjs.map
