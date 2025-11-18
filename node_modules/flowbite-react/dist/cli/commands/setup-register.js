import fs__default from 'fs/promises';
import cjson from 'comment-json';
import { packageJsonFile } from '../consts.js';
import { getPackageJson } from '../utils/get-package-json.js';

async function setupRegister() {
  try {
    const registerCommand = "flowbite-react register";
    const packageJson = await getPackageJson();
    if (!packageJson.scripts) {
      packageJson.scripts = {};
    }
    if (!packageJson.scripts.postinstall?.includes(registerCommand)) {
      console.log(`Adding postinstall register script to ${packageJsonFile}...`);
      if (packageJson.scripts.postinstall) {
        packageJson.scripts.postinstall += ` && ${registerCommand}`;
      } else {
        packageJson.scripts.postinstall = registerCommand;
      }
      await fs__default.writeFile(packageJsonFile, cjson.stringify(packageJson, null, 2));
    }
  } catch (error) {
    console.error(`Failed to setup ${packageJsonFile}:`, error);
  }
}

export { setupRegister };
//# sourceMappingURL=setup-register.js.map
