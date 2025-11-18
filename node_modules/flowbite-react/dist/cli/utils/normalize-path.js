import path__default from 'path';

function normalizeImportPath(importPath) {
  return importPath.replace(/\\/g, "/");
}
function joinNormalizedPath(...segments) {
  return normalizeImportPath(path__default.join(...segments));
}

export { joinNormalizedPath, normalizeImportPath };
//# sourceMappingURL=normalize-path.js.map
