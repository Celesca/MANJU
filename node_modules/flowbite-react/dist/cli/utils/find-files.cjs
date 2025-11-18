'use strict';

var fs = require('fs/promises');
var path = require('path');

async function findFiles({
  patterns,
  startDir = ".",
  excludeDirs = [],
  parallel = true,
  recursive = true
}) {
  const results = [];
  function matchesPattern(fileName) {
    return patterns.some((pattern) => {
      const patternParts = pattern.split("/");
      const patternFileName = patternParts[patternParts.length - 1];
      if (patternFileName && !patternFileName.includes("*")) {
        return fileName === patternFileName;
      }
      const fileExt = path.extname(fileName).slice(1);
      const patternExt = patternFileName.split(".").pop();
      return patternFileName.includes("*") && patternExt === fileExt;
    });
  }
  async function search(directory) {
    if (excludeDirs.includes(path.basename(directory))) {
      return;
    }
    try {
      const entries = await fs.readdir(directory, { withFileTypes: true });
      const tasks = entries.map(async (entry) => {
        const fullPath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
          if (recursive) {
            return search(fullPath);
          }
        } else {
          if (matchesPattern(entry.name)) {
            results.push(fullPath);
          }
        }
      });
      if (parallel) {
        await Promise.all(tasks);
      } else {
        for (const task of tasks) {
          await task;
        }
      }
    } catch (error) {
      console.error("Error reading directory:", error);
    }
  }
  await search(startDir);
  return results;
}

exports.findFiles = findFiles;
//# sourceMappingURL=find-files.cjs.map
