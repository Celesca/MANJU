import fs__default from 'fs/promises';
import { classListFilePath } from '../consts.js';

async function getClassList() {
  try {
    return JSON.parse(await fs__default.readFile(classListFilePath, "utf-8")) ?? [];
  } catch {
    return [];
  }
}

export { getClassList };
//# sourceMappingURL=get-class-list.js.map
