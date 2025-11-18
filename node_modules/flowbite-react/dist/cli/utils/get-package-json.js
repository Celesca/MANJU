import fs__default from 'fs/promises';
import cjson from 'comment-json';
import { packageJsonFile } from '../consts.js';

async function getPackageJson() {
  try {
    return cjson.parse(await fs__default.readFile(packageJsonFile, "utf-8"));
  } catch {
    console.error(`Unable to find ${packageJsonFile}.`);
    process.exit(1);
  }
}

export { getPackageJson };
//# sourceMappingURL=get-package-json.js.map
