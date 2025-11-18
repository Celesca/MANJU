'use strict';

var fs = require('fs/promises');
var path = require('path');
var compoundComponents = require('../transformers/compound-components.cjs');
var findFiles = require('../utils/find-files.cjs');

function _interopNamespaceDefault(e) {
  var n = Object.create(null);
  if (e) {
    Object.keys(e).forEach(function (k) {
      if (k !== 'default') {
        var d = Object.getOwnPropertyDescriptor(e, k);
        Object.defineProperty(n, k, d.get ? d : {
          enumerable: true,
          get: function () { return e[k]; }
        });
      }
    });
  }
  n.default = e;
  return Object.freeze(n);
}

var fs__namespace = /*#__PURE__*/_interopNamespaceDefault(fs);
var path__namespace = /*#__PURE__*/_interopNamespaceDefault(path);

const transformers = [
  compoundComponents.compoundComponentsTransformer
  // Add more transformers as needed
];
async function migrate() {
  console.log("\u{1F504} Running code transformations...");
  try {
    const files = await findFiles.findFiles({
      patterns: ["**/*.tsx", "**/*.jsx", "**/*.ts", "**/*.js"],
      excludeDirs: ["node_modules", "dist"]
    });
    let totalUpdatedFiles = 0;
    let totalSkippedFiles = 0;
    for (const transformer of transformers) {
      console.log(`
\u{1F4DD} Running ${transformer.name} transformer...`);
      let updatedFiles = 0;
      let skippedFiles = 0;
      let processedFiles = 0;
      const totalFiles = files.length;
      for (const file of files) {
        processedFiles++;
        if (processedFiles % 50 === 0 || processedFiles === totalFiles) {
          process.stdout.write(
            `\rProcessing: ${processedFiles}/${totalFiles} files (${Math.round(processedFiles / totalFiles * 100)}%)`
          );
        }
        try {
          const content = await fs__namespace.readFile(file, "utf-8");
          const result = transformer.transform(content);
          if (result.changed) {
            await fs__namespace.writeFile(file, result.content, "utf-8");
            console.log(`\u2713 Updated ${path__namespace.relative(process.cwd(), file)}`);
            updatedFiles++;
          }
        } catch (_error) {
          skippedFiles++;
          console.error(`Could not process ${path__namespace.relative(process.cwd(), file)}`);
        }
      }
      if (skippedFiles > 0) {
        console.log(`
\u2139\uFE0F  Skipped ${skippedFiles} file${skippedFiles === 1 ? "" : "s"} due to parsing errors`);
      }
      if (updatedFiles > 0) {
        console.log(
          `
\u2728 Successfully transformed ${updatedFiles} file${updatedFiles === 1 ? "" : "s"} with ${transformer.name}`
        );
      } else if (skippedFiles === 0) {
        console.log(`
\u2728 No files needed transformation with ${transformer.name}`);
      }
      totalUpdatedFiles += updatedFiles;
      totalSkippedFiles += skippedFiles;
    }
    console.log("\n\u{1F4CA} Migration Summary:");
    console.log(`Total files updated: ${totalUpdatedFiles}`);
    console.log(`Total files skipped: ${totalSkippedFiles}`);
  } catch (error) {
    console.error("Error during migration:", error);
    process.exit(1);
  }
}

exports.migrate = migrate;
//# sourceMappingURL=migrate.cjs.map
