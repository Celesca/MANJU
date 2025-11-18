import fs__default from 'fs/promises';
import { gitIgnoreFilePath, classListFile, processIdFile } from '../consts.js';

async function setupGitIgnore() {
  const content = `${classListFile}
${processIdFile}`;
  try {
    let currentContent;
    try {
      currentContent = await fs__default.readFile(gitIgnoreFilePath, "utf-8");
    } catch {
      currentContent = "";
    }
    if (currentContent.trimEnd() !== content) {
      console.log(`${currentContent ? "Updating" : "Creating"} ${gitIgnoreFilePath} file...`);
      setTimeout(() => fs__default.writeFile(gitIgnoreFilePath, content), 10);
    }
  } catch (error) {
    console.error(`Failed to update ${gitIgnoreFilePath}:`, error);
  }
}

export { setupGitIgnore };
//# sourceMappingURL=setup-gitignore.js.map
