'use strict';

var module$1 = require('module');

var _documentCurrentScript = typeof document !== 'undefined' ? document.currentScript : null;
async function getModulePackageJson(packageName) {
  try {
    const require$1 = module$1.createRequire((typeof document === 'undefined' ? require('u' + 'rl').pathToFileURL(__filename).href : (_documentCurrentScript && _documentCurrentScript.tagName.toUpperCase() === 'SCRIPT' && _documentCurrentScript.src || new URL('cli/utils/get-module-package-json.cjs', document.baseURI).href)));
    return require$1(`${packageName}/package.json`);
  } catch {
    try {
      return (await import(`${packageName}/package.json`)).default;
    } catch {
      return null;
    }
  }
}

exports.getModulePackageJson = getModulePackageJson;
//# sourceMappingURL=get-module-package-json.cjs.map
