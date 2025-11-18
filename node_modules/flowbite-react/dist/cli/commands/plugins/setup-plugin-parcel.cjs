'use strict';

var fs = require('fs/promises');
var path = require('path');
var consts = require('../../consts.cjs');
var normalizePath = require('../../utils/normalize-path.cjs');

async function setupPluginParcel(configPath) {
  try {
    const content = await fs.readFile(configPath, "utf-8");
    const parsedContent = JSON.parse(content);
    const parcelConfigDir = path.join(consts.outputDir, "parcel-config");
    const parcelReporterFile = "parcel-reporter.cjs";
    try {
      await fs.access(parcelConfigDir);
    } catch {
      console.log(`Creating ${parcelConfigDir} directory...`);
      await fs.mkdir(parcelConfigDir, { recursive: true });
    }
    const parcelConfigFilePath = path.join(parcelConfigDir, "index.json");
    const parcelConfigFileContent = {
      reporters: [`./${parcelReporterFile}`, "..."]
    };
    try {
      const fileContent = await fs.readFile(parcelConfigFilePath, "utf-8");
      const parsedContent2 = JSON.parse(fileContent);
      if (!parsedContent2.reporters?.includes(parcelConfigFileContent.reporters[0])) {
        console.log(`Updating ${parcelConfigFilePath} file...`);
        await fs.writeFile(parcelConfigFilePath, JSON.stringify(parcelConfigFileContent, null, 2), "utf-8");
      }
    } catch {
      console.log(`Creating ${parcelConfigFilePath} file...`);
      await fs.writeFile(parcelConfigFilePath, JSON.stringify(parcelConfigFileContent, null, 2), "utf-8");
    }
    const pluginImportPath = normalizePath.joinNormalizedPath(consts.pluginPath, "parcel");
    const parcelReporterFileContent = `module.exports = require("${pluginImportPath}");`;
    const parcelReporterFilePath = path.join(parcelConfigDir, parcelReporterFile);
    try {
      const fileContent = await fs.readFile(parcelReporterFilePath, "utf-8");
      if (!fileContent.includes(parcelReporterFileContent)) {
        console.log(`Updating ${parcelReporterFilePath} file...`);
        await fs.writeFile(parcelReporterFilePath, parcelReporterFileContent, "utf-8");
      }
    } catch {
      console.log(`Creating ${parcelReporterFilePath} file...`);
      await fs.writeFile(parcelReporterFilePath, parcelReporterFileContent, "utf-8");
    }
    if (!parsedContent.extends?.includes(parcelConfigFilePath)) {
      parsedContent.extends || (parsedContent.extends = []);
      parsedContent.extends.push(parcelConfigFilePath);
      console.log(`Updating ${configPath} with ${parcelConfigFilePath} config...`);
      await fs.writeFile(configPath, JSON.stringify(parsedContent, null, 2), "utf-8");
    }
  } catch (error) {
    console.error(`Failed to setup ${consts.pluginName} plugin:`, error);
  }
}

exports.setupPluginParcel = setupPluginParcel;
//# sourceMappingURL=setup-plugin-parcel.cjs.map
