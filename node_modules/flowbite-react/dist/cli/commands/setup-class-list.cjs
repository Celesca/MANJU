'use strict';

var fs = require('fs/promises');
var consts = require('../consts.cjs');

async function setupClassList() {
  try {
    await fs.access(consts.classListFilePath);
  } catch {
    console.log(`Creating ${consts.classListFilePath} file...`);
    await fs.writeFile(consts.classListFilePath, JSON.stringify([], null, 2));
  }
}

exports.setupClassList = setupClassList;
//# sourceMappingURL=setup-class-list.cjs.map
