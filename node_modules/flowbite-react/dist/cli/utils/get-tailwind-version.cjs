'use strict';

var getModulePackageJson = require('./get-module-package-json.cjs');

async function getTailwindVersion() {
  const tailwindcssPackageJson = await getModulePackageJson.getModulePackageJson("tailwindcss");
  if (!tailwindcssPackageJson) {
    throw new Error("Tailwind CSS is not installed");
  }
  const major = parseInt(tailwindcssPackageJson.version.split(".")[0], 10);
  if (major === 3 || major === 4) {
    return major;
  }
  throw new Error(`Unsupported Tailwind CSS major version: ${major}`);
}

exports.getTailwindVersion = getTailwindVersion;
//# sourceMappingURL=get-tailwind-version.cjs.map
