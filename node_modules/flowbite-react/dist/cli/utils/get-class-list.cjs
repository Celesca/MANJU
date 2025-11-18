'use strict';

var fs = require('fs/promises');
var consts = require('../consts.cjs');

async function getClassList() {
  try {
    return JSON.parse(await fs.readFile(consts.classListFilePath, "utf-8")) ?? [];
  } catch {
    return [];
  }
}

exports.getClassList = getClassList;
//# sourceMappingURL=get-class-list.cjs.map
