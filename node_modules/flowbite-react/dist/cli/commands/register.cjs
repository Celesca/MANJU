'use strict';

var child_process = require('child_process');
var fs = require('fs/promises');
var path = require('path');
var consts = require('../consts.cjs');
var getConfig = require('../utils/get-config.cjs');
var setupInit = require('./setup-init.cjs');
var setupOutputDirectory = require('./setup-output-directory.cjs');

async function register() {
  await setupOutputDirectory.setupOutputDirectory();
  const config = await getConfig.getConfig();
  await setupInit.setupInit(config);
  if (config.components.length) {
    console.warn(consts.automaticClassGenerationMessage);
  }
  const processIdFilePath = path.join(consts.outputDir, consts.processIdFile);
  try {
    const pid = await fs.readFile(processIdFilePath, "utf8");
    process.kill(parseInt(pid, 10));
    await fs.unlink(processIdFilePath);
  } catch {
  }
  try {
    const devProcess = child_process.spawn("flowbite-react", ["dev"], {
      stdio: "ignore",
      detached: true,
      shell: true
    });
    if (devProcess.pid) {
      await fs.writeFile(processIdFilePath, devProcess.pid.toString());
    }
  } catch (error) {
    console.error("Failed to register flowbite-react", error);
  }
  process.exit();
}

exports.register = register;
//# sourceMappingURL=register.cjs.map
