'use strict';

var fs = require('fs/promises');
var consts = require('../consts.cjs');

async function setupGitIgnore() {
  const content = `${consts.classListFile}
${consts.processIdFile}`;
  try {
    let currentContent;
    try {
      currentContent = await fs.readFile(consts.gitIgnoreFilePath, "utf-8");
    } catch {
      currentContent = "";
    }
    if (currentContent.trimEnd() !== content) {
      console.log(`${currentContent ? "Updating" : "Creating"} ${consts.gitIgnoreFilePath} file...`);
      setTimeout(() => fs.writeFile(consts.gitIgnoreFilePath, content), 10);
    }
  } catch (error) {
    console.error(`Failed to update ${consts.gitIgnoreFilePath}:`, error);
  }
}

exports.setupGitIgnore = setupGitIgnore;
//# sourceMappingURL=setup-gitignore.cjs.map
