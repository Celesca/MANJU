import { ensureTailwind } from './ensure-tailwind.js';
import { installPackage } from './install.js';
import { setupClassList } from './setup-class-list.js';
import { setupConfig } from './setup-config.js';
import { setupGitIgnore } from './setup-gitignore.js';
import { setupInit } from './setup-init.js';
import { setupOutputDirectory } from './setup-output-directory.js';
import { setupPlugin } from './setup-plugin.js';
import { setupRegister } from './setup-register.js';
import { setupTailwind } from './setup-tailwind.js';
import { setupVSCode } from './setup-vscode.js';

async function init() {
  try {
    await ensureTailwind();
    await installPackage();
    await setupOutputDirectory();
    await setupGitIgnore();
    await setupClassList();
    const config = await setupConfig();
    await setupInit(config);
    await setupVSCode();
    await setupTailwind();
    const hasBundler = await setupPlugin();
    if (!hasBundler) {
      await setupRegister();
    }
    console.log("\n\u2705 Flowbite React has been successfully initialized!");
  } catch (error) {
    console.error("An error occurred during initialization:", error);
    process.exit(1);
  }
}

export { init };
//# sourceMappingURL=init.js.map
