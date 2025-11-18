import fs__default from 'fs/promises';
import { classListFilePath } from '../consts.js';

async function setupClassList() {
  try {
    await fs__default.access(classListFilePath);
  } catch {
    console.log(`Creating ${classListFilePath} file...`);
    await fs__default.writeFile(classListFilePath, JSON.stringify([], null, 2));
  }
}

export { setupClassList };
//# sourceMappingURL=setup-class-list.js.map
