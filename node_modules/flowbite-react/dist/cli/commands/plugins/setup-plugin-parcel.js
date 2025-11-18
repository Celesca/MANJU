import fs__default from 'fs/promises';
import path__default from 'path';
import { outputDir, pluginPath, pluginName } from '../../consts.js';
import { joinNormalizedPath } from '../../utils/normalize-path.js';

async function setupPluginParcel(configPath) {
  try {
    const content = await fs__default.readFile(configPath, "utf-8");
    const parsedContent = JSON.parse(content);
    const parcelConfigDir = path__default.join(outputDir, "parcel-config");
    const parcelReporterFile = "parcel-reporter.cjs";
    try {
      await fs__default.access(parcelConfigDir);
    } catch {
      console.log(`Creating ${parcelConfigDir} directory...`);
      await fs__default.mkdir(parcelConfigDir, { recursive: true });
    }
    const parcelConfigFilePath = path__default.join(parcelConfigDir, "index.json");
    const parcelConfigFileContent = {
      reporters: [`./${parcelReporterFile}`, "..."]
    };
    try {
      const fileContent = await fs__default.readFile(parcelConfigFilePath, "utf-8");
      const parsedContent2 = JSON.parse(fileContent);
      if (!parsedContent2.reporters?.includes(parcelConfigFileContent.reporters[0])) {
        console.log(`Updating ${parcelConfigFilePath} file...`);
        await fs__default.writeFile(parcelConfigFilePath, JSON.stringify(parcelConfigFileContent, null, 2), "utf-8");
      }
    } catch {
      console.log(`Creating ${parcelConfigFilePath} file...`);
      await fs__default.writeFile(parcelConfigFilePath, JSON.stringify(parcelConfigFileContent, null, 2), "utf-8");
    }
    const pluginImportPath = joinNormalizedPath(pluginPath, "parcel");
    const parcelReporterFileContent = `module.exports = require("${pluginImportPath}");`;
    const parcelReporterFilePath = path__default.join(parcelConfigDir, parcelReporterFile);
    try {
      const fileContent = await fs__default.readFile(parcelReporterFilePath, "utf-8");
      if (!fileContent.includes(parcelReporterFileContent)) {
        console.log(`Updating ${parcelReporterFilePath} file...`);
        await fs__default.writeFile(parcelReporterFilePath, parcelReporterFileContent, "utf-8");
      }
    } catch {
      console.log(`Creating ${parcelReporterFilePath} file...`);
      await fs__default.writeFile(parcelReporterFilePath, parcelReporterFileContent, "utf-8");
    }
    if (!parsedContent.extends?.includes(parcelConfigFilePath)) {
      parsedContent.extends || (parsedContent.extends = []);
      parsedContent.extends.push(parcelConfigFilePath);
      console.log(`Updating ${configPath} with ${parcelConfigFilePath} config...`);
      await fs__default.writeFile(configPath, JSON.stringify(parsedContent, null, 2), "utf-8");
    }
  } catch (error) {
    console.error(`Failed to setup ${pluginName} plugin:`, error);
  }
}

export { setupPluginParcel };
//# sourceMappingURL=setup-plugin-parcel.js.map
