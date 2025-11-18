'use strict';

var getPackageJson = require('../utils/get-package-json.cjs');

async function ensureTailwind() {
  const packageJson = await getPackageJson.getPackageJson();
  const packageName = "tailwindcss";
  if (!(packageJson?.dependencies?.[packageName] || packageJson?.devDependencies?.[packageName])) {
    console.error("Install Tailwind CSS first.\n\nSee: https://tailwindcss.com/docs/installation");
    process.exit(1);
  }
}

exports.ensureTailwind = ensureTailwind;
//# sourceMappingURL=ensure-tailwind.cjs.map
