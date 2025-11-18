'use strict';

var commands = require('package-manager-detector/commands');
var detect = require('package-manager-detector/detect');
var execCommand = require('../utils/exec-command.cjs');
var getModulePackageJson = require('../utils/get-module-package-json.cjs');
var getPackageJson = require('../utils/get-package-json.cjs');

async function installPackage() {
  const packageName = "flowbite-react";
  try {
    let pm = await detect.detect();
    if (!pm) {
      console.error("Could not detect package manager");
    }
    pm ?? (pm = { agent: "npm", name: "npm" });
    const packageJson = await getPackageJson.getPackageJson();
    const currentPackage = await getModulePackageJson.getModulePackageJson(packageName);
    if (currentPackage && (packageJson?.dependencies?.[packageName] || packageJson?.devDependencies?.[packageName])) {
      if (currentPackage.version.localeCompare("0.11", void 0, { numeric: true }) < 0) {
        console.log(
          "The current version of flowbite-react is below 0.11.x, which is the version with the new engine and CLI."
        );
        const { command: command2 = "", args: args2 } = commands.resolveCommand(pm.agent, "add", [`${packageName}@latest`]) ?? {};
        console.log(`Updating ${packageName} to latest version using ${pm.name}...`);
        await execCommand.execCommand(withWindowsSupport(command2), args2);
      }
      return;
    }
    const { command = "", args } = commands.resolveCommand(pm.agent, "add", [packageName]) ?? {};
    console.log(`Installing ${packageName} using ${pm.name}...`);
    await execCommand.execCommand(withWindowsSupport(command), args);
  } catch (error) {
    console.error(`Failed to install ${packageName}:`, error);
  }
}
function withWindowsSupport(command) {
  return process.platform === "win32" ? `${command}.cmd` : command;
}

exports.installPackage = installPackage;
//# sourceMappingURL=install.cjs.map
