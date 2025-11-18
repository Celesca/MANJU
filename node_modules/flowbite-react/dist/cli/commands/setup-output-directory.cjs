'use strict';

var fs = require('fs/promises');
var consts = require('../consts.cjs');

async function setupOutputDirectory() {
  try {
    await fs.access(consts.outputDir);
  } catch {
    console.log(`Creating ${consts.outputDir} directory...`);
    await fs.mkdir(consts.outputDir, { recursive: true });
  }
}

exports.setupOutputDirectory = setupOutputDirectory;
//# sourceMappingURL=setup-output-directory.cjs.map
