'use strict';

var fs = require('fs/promises');
var cjson = require('comment-json');
var consts = require('../consts.cjs');
var getPackageJson = require('../utils/get-package-json.cjs');

async function setupRegister() {
  try {
    const registerCommand = "flowbite-react register";
    const packageJson = await getPackageJson.getPackageJson();
    if (!packageJson.scripts) {
      packageJson.scripts = {};
    }
    if (!packageJson.scripts.postinstall?.includes(registerCommand)) {
      console.log(`Adding postinstall register script to ${consts.packageJsonFile}...`);
      if (packageJson.scripts.postinstall) {
        packageJson.scripts.postinstall += ` && ${registerCommand}`;
      } else {
        packageJson.scripts.postinstall = registerCommand;
      }
      await fs.writeFile(consts.packageJsonFile, cjson.stringify(packageJson, null, 2));
    }
  } catch (error) {
    console.error(`Failed to setup ${consts.packageJsonFile}:`, error);
  }
}

exports.setupRegister = setupRegister;
//# sourceMappingURL=setup-register.cjs.map
