import fs__default from 'fs/promises';
import { outputDir } from '../consts.js';

async function setupOutputDirectory() {
  try {
    await fs__default.access(outputDir);
  } catch {
    console.log(`Creating ${outputDir} directory...`);
    await fs__default.mkdir(outputDir, { recursive: true });
  }
}

export { setupOutputDirectory };
//# sourceMappingURL=setup-output-directory.js.map
