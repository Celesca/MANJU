import { spawn } from 'child_process';
import fs__default from 'fs/promises';
import path__default from 'path';
import { automaticClassGenerationMessage, outputDir, processIdFile } from '../consts.js';
import { getConfig } from '../utils/get-config.js';
import { setupInit } from './setup-init.js';
import { setupOutputDirectory } from './setup-output-directory.js';

async function register() {
  await setupOutputDirectory();
  const config = await getConfig();
  await setupInit(config);
  if (config.components.length) {
    console.warn(automaticClassGenerationMessage);
  }
  const processIdFilePath = path__default.join(outputDir, processIdFile);
  try {
    const pid = await fs__default.readFile(processIdFilePath, "utf8");
    process.kill(parseInt(pid, 10));
    await fs__default.unlink(processIdFilePath);
  } catch {
  }
  try {
    const devProcess = spawn("flowbite-react", ["dev"], {
      stdio: "ignore",
      detached: true,
      shell: true
    });
    if (devProcess.pid) {
      await fs__default.writeFile(processIdFilePath, devProcess.pid.toString());
    }
  } catch (error) {
    console.error("Failed to register flowbite-react", error);
  }
  process.exit();
}

export { register };
//# sourceMappingURL=register.js.map
