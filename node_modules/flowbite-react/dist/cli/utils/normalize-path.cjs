'use strict';

var path = require('path');

function normalizeImportPath(importPath) {
  return importPath.replace(/\\/g, "/");
}
function joinNormalizedPath(...segments) {
  return normalizeImportPath(path.join(...segments));
}

exports.joinNormalizedPath = joinNormalizedPath;
exports.normalizeImportPath = normalizeImportPath;
//# sourceMappingURL=normalize-path.cjs.map
