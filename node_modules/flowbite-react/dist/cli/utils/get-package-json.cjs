'use strict';

var fs = require('fs/promises');
var cjson = require('comment-json');
var consts = require('../consts.cjs');

async function getPackageJson() {
  try {
    return cjson.parse(await fs.readFile(consts.packageJsonFile, "utf-8"));
  } catch {
    console.error(`Unable to find ${consts.packageJsonFile}.`);
    process.exit(1);
  }
}

exports.getPackageJson = getPackageJson;
//# sourceMappingURL=get-package-json.cjs.map
