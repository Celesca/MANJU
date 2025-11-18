'use strict';

var ensureTailwind = require('./ensure-tailwind.cjs');
var install = require('./install.cjs');
var setupClassList = require('./setup-class-list.cjs');
var setupConfig = require('./setup-config.cjs');
var setupGitignore = require('./setup-gitignore.cjs');
var setupInit = require('./setup-init.cjs');
var setupOutputDirectory = require('./setup-output-directory.cjs');
var setupPlugin = require('./setup-plugin.cjs');
var setupRegister = require('./setup-register.cjs');
var setupTailwind = require('./setup-tailwind.cjs');
var setupVscode = require('./setup-vscode.cjs');

async function init() {
  try {
    await ensureTailwind.ensureTailwind();
    await install.installPackage();
    await setupOutputDirectory.setupOutputDirectory();
    await setupGitignore.setupGitIgnore();
    await setupClassList.setupClassList();
    const config = await setupConfig.setupConfig();
    await setupInit.setupInit(config);
    await setupVscode.setupVSCode();
    await setupTailwind.setupTailwind();
    const hasBundler = await setupPlugin.setupPlugin();
    if (!hasBundler) {
      await setupRegister.setupRegister();
    }
    console.log("\n\u2705 Flowbite React has been successfully initialized!");
  } catch (error) {
    console.error("An error occurred during initialization:", error);
    process.exit(1);
  }
}

exports.init = init;
//# sourceMappingURL=init.cjs.map
